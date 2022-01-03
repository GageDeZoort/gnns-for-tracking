# System
import os
import sys
import argparse
import logging
import multiprocessing as mp
from functools import partial
from collections import Counter

# Externals
import yaml
import pickle
import numpy as np
import pandas as pd
import trackml.dataset
import time
from torch_geometric.data import Data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('prepare.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/prepare_trackml.yaml')
    add_arg('--n-workers', type=int, default=1)
    add_arg('--task', type=int, default=0)
    add_arg('--n-tasks', type=int, default=1)
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--show-config', action='store_true')
    add_arg('--interactive', action='store_true')
    add_arg('--start-evtid', type=int, default=1000)
    add_arg('--end-evtid', type=int, default=3000)
    return parser.parse_args()

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def calc_eta(r, z):
    """Computes pseudorapidity
       (https://en.wikipedia.org/wiki/Pseudorapidity)
    """
    theta = np.arctan2(r, z)
    return -1. * np.log(np.tan(theta / 2.))

def select_edges(hits1, hits2, layer1, layer2, 
                 phi_slope_max, z0_max, module_map=[],
                 use_module_map=False):

    # start with all possible pairs of hits
    keys = ['evtid', 'r', 'phi', 'z', 'module_id', 'overlapped']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))

    # compute geometric features of the line through each hit pair
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    eta_1 = calc_eta(hit_pairs.r_1, hit_pairs.z_1)
    eta_2 = calc_eta(hit_pairs.r_2, hit_pairs.z_2)
    deta = eta_2 - eta_1
    dR = np.sqrt(deta**2 + dphi**2)
    
    # phi_slope and z0 used to filter spurious edges
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
    
    # apply the intersecting line cut 
    intersected_layer = dr.abs() < -1 
    # 0th barrel layer to left EC or right EC
    if (layer1 == 0) and (layer2 == 11 or layer2 == 4): 
        z_coord = 71.56298065185547 * dz/dr + z0
        intersected_layer = np.logical_and(z_coord > -490.975, 
                                           z_coord < 490.975)
    # 1st barrel layer to the left EC or right EC
    if (layer1 == 1) and (layer2 == 11 or layer2 == 4): 
        z_coord = 115.37811279296875 * dz / dr + z0
        intersected_layer = np.logical_and(z_coord > -490.975, 
                                           z_coord < 490.975)
        
    # mask edges not in the module map
    if (use_module_map):
        mid1 = hit_pairs.module_id_1.values
        mid2 = hit_pairs.module_id_2.values
        in_module_map = module_map[mid1, mid2]
    else: in_module_map = np.ones(z0.shape, dtype=bool)

    # filter edges according to selection criteria
    good_edge_mask = ((phi_slope.abs() < phi_slope_max) & # geometric
                      (z0.abs() < z0_max) &               # geometric
                      (intersected_layer == False) &      # geometric
                      (in_module_map))                    # data-driven
    
    # store edges (in COO format) and geometric edge features 
    selected_edges = {'edges': hit_pairs[['index_1', 'index_2']][good_edge_mask],
                      'dr': dr[good_edge_mask],    
                      'dphi': dphi[good_edge_mask], 
                      'dz': dz[good_edge_mask],
                      'dR': dR[good_edge_mask]}
    
    return selected_edges 

def check_truth_labels(hits, edges, y, particle_ids):
    """ Corrects for extra edges surviving the barrel intersection
        cut, i.e. for each particle counts the number of extra 
        "transition edges" crossing from a barrel layer to an 
        innermost endcap slayer; the sum is n_incorrect
        - [edges] = n_edges x 2
        - [y] = n_edges
        - [particle_ids] = n_edges
    """
    # layer indices for barrel-to-endcap edges
    barrel_to_endcaps = {(0,4), (1,4), (2,4),    # barrel to l-EC
                         (0,11), (1,11), (2,11)} # barrel to r-EC
    
    # group hits by particle id, get layer indices
    hits_by_particle = hits.groupby('particle_id')
    layers_1 = hits.layer.loc[edges.index_1].values
    layers_2 = hits.layer.loc[edges.index_2].values

    # loop over particle_id, particle_hits, 
    # count extra transition edges as n_incorrect
    n_incorrect = 0
    for p, particle_hits in hits_by_particle:
        particle_hit_ids = particle_hits['hit_id'].values
        
        # grab true segment indices for particle p
        relevant_indices = ((particle_ids==p) & (y==1))
        
        # get layers connected by particle's edges
        particle_l1 = layers_1[relevant_indices]
        particle_l2 = layers_2[relevant_indices]
        layer_pairs = set(zip(particle_l1, particle_l2))
        
        # count the number of transition edges
        transition_edges = layer_pairs.intersection(barrel_to_endcaps)
        if (len(transition_edges) > 1):
            n_incorrect += 1
            
    if (n_incorrect > 0):
        logging.info(f'incorrectly-labeled edges: {n_incorrect}')
    
    return n_incorrect

def construct_graph(hits, layer_pairs, phi_slope_max, z0_max,
                    feature_names, feature_scale, evtid="-1",
                    module_map={}, s=(-1,-1), use_module_map=False):
    """ Loops over hits in layer pairs and extends edges
        between them based on geometric and/or data-driven
        constraints. 
    """
    # loop over layer pairs, assign edges between their hits
    groups = hits.groupby('layer')
    edges, dr, dphi, dz, dR = [], [], [], [], []
    for (layer1, layer2) in layer_pairs:
        try:
            hits1 = groups.get_group(layer1)
            hits2 = groups.get_group(layer2)
        except KeyError as e: # skip if layer is empty
            continue
            
        # assign edges based on geometric and data-driven constraints
        selected_edges = select_edges(hits1, hits2, layer1, layer2,
                                      phi_slope_max, z0_max,  # geometric 
                                      module_map=module_map,  # data-driven
                                      use_module_map=use_module_map)  
        edges.append(selected_edges['edges'])
        dr.append(selected_edges['dr'])
        dphi.append(selected_edges['dphi'])
        dz.append(selected_edges['dz'])
        dR.append(selected_edges['dR'])
    
    # if edges were reconstructed, concatenate edge 
    # attributes and indices across all layer pairs 
    if len(edges) > 0:
        edges = pd.concat(edges)
        dr, dphi = pd.concat(dr), pd.concat(dphi)
        dz, dR = pd.concat(dz), pd.concat(dR)
    else: # if no edges were reconstructed, return empty graph 
        edges = np.array([])
        dr, dphi = np.array([]), np.array([])
        dz, dR = np.array([]), np.array([])
        x = (hits[feature_names].values / feature_scale).astype(np.float32)
        return {'x': x, 'hit_id': hits['hit_id'],
                'particle_id': hits['particle_id'],
                'edge_index': np.array([[],[]]),
                'edge_attr': np.array([[],[],[],[]]),
                'y': [], 's': s, 'n_incorrect': 0,
                'edge_hit_id': []}
    
    # prepare the graph matrices
    n_nodes = hits.shape[0]
    n_edges = edges.shape[0]
    
    # select and scale relevant features
    x = (hits[feature_names].values / feature_scale).astype(np.float32)
    edge_attr = np.stack((dr/feature_scale[0], 
                          dphi/feature_scale[1], 
                          dz/feature_scale[2], 
                          dR))
    y = np.zeros(n_edges, dtype=np.float32)

    # use a series to map hit label-index onto positional-index.
    node_idx = pd.Series(np.arange(n_nodes), index=hits.index)
    edge_start = node_idx.loc[edges.index_1].values
    edge_end = node_idx.loc[edges.index_2].values
    edge_index = np.stack((edge_start, edge_end))

    # fill the edge, particle labels
    # true edges have the same pid, ignore noise (pid=0)
    pid1 = hits.particle_id.loc[edges.index_1].values
    pid2 = hits.particle_id.loc[edges.index_2].values
    y[:] = ((pid1 == pid2) & (pid1>0) & (pid2>0)) 
    n_incorrect = check_truth_labels(hits, edges, y, pid1)
    
    # check if edges are both in overlap region
    id1 = hits.hit_id.loc[edges.index_1].values
    id2 = hits.hit_id.loc[edges.index_2].values
    edge_hit_id = np.stack((id1, id2), axis=-1)

    return {'x': x, 'hit_id': hits['hit_id'],
            'particle_id': hits['particle_id'],
            'edge_index': edge_index, 'edge_attr': edge_attr, 
            'y': y, 's': s, 'n_incorrect': n_incorrect,
            'edge_hit_id': np.transpose(edge_hit_id)}

def select_hits(hits, truth, particles, pt_min=0, endcaps=False, 
                remove_noise=False, remove_duplicates=False):
     
    # Barrel volume and layer ids
    vlids = [(8,2), # 0 
             (8,4), # 1
             (8,6), # 2
             (8,8)] # 3
    if (endcaps): 
        vlids.extend([(7,14), # 4 
                      (7,12), # 5
                      (7,10), # 6
                      (7,8),  # 7
                      (7,6),  # 8
                      (7,4),  # 9
                      (7,2),  # 10
                      (9,2),  # 11
                      (9,4),  # 12
                      (9,6),  # 13
                      (9,8),  # 14
                      (9,10), # 15
                      (9,12), # 16
                      (9,14), # 17
                     ])
    n_det_layers = len(vlids)
    
    # Select barrel layers and assign convenient layer number [0-9]
    vlid_groups = hits.groupby(['volume_id', 'layer_id'])
    hits = pd.concat([vlid_groups.get_group(vlids[i]).assign(layer=i)
                      for i in range(n_det_layers)])
    
    # Calculate particle transverse momentum
    particles['pt'] = np.sqrt(particles.px**2 + particles.py**2)
    particles['eta_pt'] = calc_eta(particles.pt, particles.pz)
    
    # True particle selection.
    particles = particles[particles.pt > pt_min]
    truth_noise = truth[['hit_id', 'particle_id']][truth.particle_id==0]
    truth_noise['pt'] = 0
    truth = (truth[['hit_id', 'particle_id']]
             .merge(particles[['particle_id', 'pt', 'eta_pt']], on='particle_id'))

    # optionally add noise 
    if (not remove_noise): 
        truth = truth.append(truth_noise)

    # calculate derived hits variables
    hits['r'] = np.sqrt(hits.x**2 + hits.y**2)
    hits['phi'] = np.arctan2(hits.y, hits.x)
    hits['eta'] = calc_eta(hits.r, hits.z)
    
    # select the data columns we need
    hits = (hits[['hit_id', 'r', 'phi', 'eta', 'z', 'layer', 'module_id']]
            .merge(truth[['hit_id', 'particle_id', 'pt', 'eta_pt']], on='hit_id'))
    
    # optionally remove duplicates
    if (remove_duplicates):
        noise_hits = hits[hits.particle_id==0]
        particle_hits = hits[hits.particle_id!=0]
        particle_hits = particle_hits.loc[particle_hits.groupby(['particle_id', 'layer']).r.idxmin()]
        hits = particle_hits.append(noise_hits)
        
    # relabel particle IDs in [1:n_particles]
    particles = particles[particles.particle_id.isin(pd.unique(hits.particle_id))]
    particle_id_map = {p: i+1 for i, p in enumerate(particles['particle_id'].values)}
    particle_id_map[0] = 0
    particles = particles.assign(particle_id=particles['particle_id'].map(particle_id_map))
    hits = hits.assign(particle_id=hits['particle_id'].map(particle_id_map))
    return hits, particles

def get_particle_properties(hits_by_particle, valid_connections, debug=False):
    """ Calculates the following truth quantities per particle:
         - n_track_segs: number of track segments generated
         - reconstructable: true if particle doesn't skip a layer
         - pt: particle transverse momentum [GeV]
         - eta: pseudorapidity w.r.t. transverse and longitudinal momentum
    """
    # loop over particle_ids and corresponding particle hits
    n_track_segs, reconstructable = {}, {}
    pt, eta = {}, {}
    for particle_id, particle_hits in hits_by_particle:
        
        # noise isn't reconstructable, store 0s
        if (particle_id==0): 
            reconstructable[particle_id] = 0
            pt[particle_id] = 0
            eta[particle_id] = 0
            n_track_segs[particle_id] = 0
            continue
            
        # store pt and eta 
        pt[particle_id] = particle_hits.pt.values[0]
        eta[particle_id] = particle_hits.eta_pt.values[0]
        
        # store hit multiplicity per layer 
        layers_hit = particle_hits.layer.values
        hits_per_layer = Counter(layers_hit) 
        layers = np.unique(layers_hit)
        
        # single-hits aren't reconstructable
        if (len(layers)==1): 
            reconstructable[particle_id] = 0
            n_track_segs[particle_id] = 0
            continue
        
        # all edges must be valid for a reconstructable particle
        layer_pairs = set(zip(layers[:-1], layers[1:]))
        reconstructable[particle_id] = layer_pairs.issubset(valid_connections)
        
        # total number of track segments produced by particle 
        good_layer_pairs = layer_pairs.intersection(valid_connections)
        count = 0
        for good_lp in good_layer_pairs:
            count += hits_per_layer[good_lp[0]] * hits_per_layer[good_lp[1]]
        n_track_segs[particle_id] = count
        
        if debug and (particle_id%100==0):
            print('Test Hit Pattern:', layers_hit)
            print(' - layer pairs:', layer_pairs)
            print(' - reconstructable:', reconstructable[particle_id])
            print(' - n_track_segs:', n_track_segs[particle_id])
            print(' - pt', pt[particle_id])
            print(' - eta', eta[particle_id])
        
    return {'pt': pt, 'eta': eta, 'n_track_segs': n_track_segs, 
            'reconstructable': reconstructable}

def get_n_track_segs(hits_by_particle, valid_connections):
    """ Calculates the number of track segments present in 
        a subset of hits generated by a particle
        (used for analyzing efficiency per sector)
    """
    # loop over particle_ids and corresponding particle hits
    n_track_segs = {}
    for particle_id, particle_hits in hits_by_particle:
        
        # noise doesn't produce true edges
        if (particle_id==0): 
            n_track_segs[particle_id] = 0
            continue
            
        # store hit multiplicity per layer 
        layers_hit = particle_hits.layer.values
        hits_per_layer = Counter(layers_hit) 
        layers = np.unique(layers_hit)
        
        # single-hits don't produce truth edges
        if (len(layers)==1): 
            n_track_segs[particle_id] = 0
            continue
        
        # all edges must be valid for a reconstructable particle
        layer_pairs = set(zip(layers[:-1], layers[1:]))
        
        # total number of true edges produced by particle 
        good_layer_pairs = layer_pairs.intersection(valid_connections)
        count = 0
        for good_lp in good_layer_pairs:
            count += hits_per_layer[good_lp[0]] * hits_per_layer[good_lp[1]]
        n_track_segs[particle_id] = count
        
    return n_track_segs

def split_detector_sectors(hits, phi_edges, eta_edges, verbose=False, 
                           phi_overlap=0.1, eta_overlap=0.1):

    """Split hits according to provided phi and eta boundaries."""
    hits_sectors = {}
    sector_info = {}
    for i in range(len(phi_edges) - 1):
        phi_min = phi_edges[i]
        phi_max = phi_edges[i+1]
        
        # simple check that we're in the phi sector
        in_phi_sector = ((hits.phi > phi_min) & 
                         (hits.phi < phi_max))
        
        # select hits in this phi sector
        phi_hits = hits[in_phi_sector]
        phi_hits = phi_hits.assign(old_phi=phi_hits.phi)
        
        # deterime upper and lower overlap regions
        lower_overlap = ((hits.phi > (phi_min - phi_overlap)) & 
                         (hits.phi < phi_min)) 
        upper_overlap = ((hits.phi < (phi_max + phi_overlap)) & 
                         (hits.phi > phi_max))
        
        # interior overlap accounts for where other sectors will intrude
        interior_overlap = (((phi_hits.phi > phi_min) & 
                             (phi_hits.phi < (phi_min+phi_overlap))) | 
                            ((phi_hits.phi > (phi_max-phi_overlap)) & 
                             (phi_hits.phi < phi_max)))
        
        # special case: regions bounded at +/-pi branch cut 
        if (abs(phi_min-(-np.pi))<0.01):
            lower_overlap = ((hits.phi > (np.pi-phi_overlap)) & 
                             (hits.phi < np.pi))
            upper_overlap = ((hits.phi < (phi_max + phi_overlap)) & 
                             (hits.phi > phi_max))
        if (abs(phi_max-np.pi) < 0.01):
            lower_overlap = ((hits.phi > (phi_min - phi_overlap)) &
                             (hits.phi < phi_min)) 
            upper_overlap = ((hits.phi > -np.pi) &
                             (hits.phi < (-np.pi+phi_overlap)))

        # select hits in overlapped region
        lower_overlap_hits = hits[lower_overlap]
        upper_overlap_hits = hits[upper_overlap]
        lower_phi = lower_overlap_hits.phi
        lower_overlap_hits = lower_overlap_hits.assign(old_phi=lower_phi)
        upper_phi = upper_overlap_hits.phi
        upper_overlap_hits = upper_overlap_hits.assign(old_phi=upper_phi)
        
        # adjust values across +/-pi branch cut 
        if (abs(phi_min-(-np.pi)) < 0.01):
            new_lower_phi = -2*np.pi + lower_overlap_hits.phi
            lower_overlap_hits = lower_overlap_hits.assign(phi=new_lower_phi)
        if (abs(phi_max-np.pi) < 0.01):
            new_upper_phi = 2*np.pi + upper_overlap_hits.phi
            upper_overlap_hits = upper_overlap_hits.assign(phi=new_upper_phi)
        
        # center these hits on phi=0
        phi_hits = phi_hits.assign(overlapped=np.zeros(len(phi_hits), dtype=bool))
        phi_hits.loc[interior_overlap, 'overlapped'] = True
        lower_overlap_hits = lower_overlap_hits.assign(overlapped=np.ones(len(lower_overlap_hits), dtype=bool))
        upper_overlap_hits = upper_overlap_hits.assign(overlapped=np.ones(len(upper_overlap_hits), dtype=bool))
        phi_hits = phi_hits.append(lower_overlap_hits)
        phi_hits = phi_hits.append(upper_overlap_hits)
        
        centered_phi = phi_hits.phi - (phi_min + phi_max) / 2.
        phi_hits = phi_hits.assign(phi=centered_phi, phi_sector=i)
        for j in range(len(eta_edges) - 1):
            eta_min = eta_edges[j] 
            eta_max = eta_edges[j+1] 
            
            # select hits in this eta sector
            eta = calc_eta(phi_hits.r, phi_hits.z)
            in_eta_overlap = (((eta > eta_min-eta_overlap) & (eta < eta_min)) | 
                              ((eta < eta_max+eta_overlap) & (eta > eta_max)))
            interior_eta_overlap = (((eta > eta_min) & 
                                     (eta < (eta_min+eta_overlap))) | 
                                    ((eta > (eta_max-eta_overlap)) & 
                                     (eta < eta_max)))
            sec_hits = phi_hits[(in_eta_overlap | 
                                ((eta > eta_min) & 
                                 (eta < eta_max)))]
            sec_hits = sec_hits.assign(overlapped=(sec_hits.overlapped & 
                                                   (in_eta_overlap | interior_eta_overlap)))
    
            # label hits by tuple s = (eta_sector, phi_sector)
            hits_sectors[(j,i)] = sec_hits.assign(eta_sector=j)
                         
            # store eta and phi ranges per sector
            sector_info[(j,i)] = {'eta_range': [eta_min, eta_max],
                                  'phi_range': [phi_min, phi_max],
                                  'eta_overlap': eta_overlap,
                                  'phi_overlap': phi_overlap}
            if verbose:
                logging.info(f"Sector ({i},{j}):\n" + 
                             f"...eta_range=({eta_min:.3f},{eta_max:.3f})\n"
                             f"...phi_range=({phi_min:.3f},{phi_max:.3f})")
    
    return hits_sectors, sector_info

def graph_summary(evtid, sectors, particle_properties, 
                  sector_info, print_per_layer=False):
    """ Calculates per-sector and per-graph summary stats
        and returns a dictionary for subsequent analysis
         - total_track_segs: # track segments (true edges) possible 
         - total_nodes: # nodes present in graph / sector
         - total_edges: # edges present in graph / sector
         - total_true: # true edges present in graph / sector
         - total_false # false edges present in graph / sector
         - boundary_fraction: fraction of track segs lost between sectors
         
    """
   
    # truth number of track segments possible
    track_segs = particle_properties['n_track_segs'].values()
    total_track_segs = np.sum(list(track_segs))
    total_track_segs_sectored = 0
    
    # reconstructed quantities per graph
    total_nodes, total_edges = 0, 0
    total_true, total_false = 0, 0

    # helper function for division by 0
    def div(a,b):
        return float(a)/b if b else 0
    
    # loop over graph sectors and compile statistics
    sector_stats = {}
    total_possible_per_s = 0
    all_edges, all_truth = [], []
    for i, sector in enumerate(sectors):
        
        # get information about the graph's sector
        s = sector['s'] # s = sector label
        sector_ranges = sector_info[s]

        # catalogue all edges by id
        all_edges.extend(np.transpose(sector['edge_hit_id']).tolist())
        all_truth.extend(sector['y'].tolist())

        # calculate graph properties
        n_nodes = sector['x'].shape[0]
        total_nodes += n_nodes
        # correct n_edges for multiple transition edges
        # (see check_truth_labels()) 
        n_true = np.sum(sector['y']) - sector['n_incorrect']
        total_true += n_true
        n_false = np.sum(sector['y']==0)
        total_false += n_false
        n_edges = len(sector['y'])
        total_edges += n_edges
        
        # calculate track segments in sector
        n_track_segs_per_pid = particle_properties['n_track_segs_per_s'][s]
        n_track_segs = np.sum(list(n_track_segs_per_pid.values()))
        total_track_segs_sectored += n_track_segs
        
        # estimate purity in each sector
        sector_stats[i] = {'eta_range': sector_ranges['eta_range'],
                           'phi_range': sector_ranges['phi_range'],
                           'n_nodes': n_nodes, 'n_edges': n_edges,
                           'purity': div(n_true, n_edges),
                           'efficiency': div(n_true, n_track_segs)}

    # count duplicated true and false edges
    all_true_edges = [frozenset(ae) for i, ae in enumerate(all_edges) 
                      if all_truth[i]==1]
    all_false_edges = [frozenset(ae) for i, ae in enumerate(all_edges)
                       if all_truth[i]==0]
    te_counts = np.array(list(Counter(all_true_edges).values()))
    fe_counts = np.array(list(Counter(all_false_edges).values()))
    te_excess = np.sum(te_counts[te_counts > 1] - 1)
    fe_excess = np.sum(fe_counts[fe_counts > 1] - 1)

    # proportion of true edges to all possible track segments
    efficiency = div(total_true-te_excess, total_track_segs)
    # proportion of true edges to total reconstructed edges
    purity = div(total_true, total_edges)
    purity_corr = div(total_true - te_excess,
                      total_edges - te_excess - fe_excess)
    # proportion of true track segments lost in sector boundaries
    boundary_fraction = div(total_track_segs - total_track_segs_sectored, 
                            total_track_segs)
    
    logging.info(f'Event {evtid}, graph summary statistics\n' + 
                 f'...total nodes: {total_nodes}\n' +
                 f'...total edges: {total_edges}\n' + 
                 f'...efficiency: {efficiency:.5f}\n' +
                 f'...purity: {purity:.5f}\n' + 
                 f'...purity_corr: {purity_corr:.5f}\n' +
                 f'...duplicated true edges: {te_excess}\n' + 
                 f'...duplicated false edges: {fe_excess}\n' +
                 f'...boundary edge fraction: {boundary_fraction:.5f}')

    return {'n_nodes': total_nodes, 'n_edges': total_edges,
            'efficiency': efficiency, 'purity': purity,
            'boundary_fraction': boundary_fraction,
            'sector_stats': sector_stats}

def process_event(prefix, output_dir, module_map, pt_min, 
                  n_eta_sectors, n_phi_sectors,
                  eta_range, phi_range, phi_slope_max, z0_max,
                  endcaps, remove_noise, remove_duplicates,
                  use_module_map, phi_overlap, eta_overlap):
    
    # define valid layer pair connections
    layer_pairs = [(0,1), (1,2), (2,3)] # barrel-barrel
    if endcaps:
        layer_pairs.extend([(0,4), (1,4), (2,4),  # barrel-LEC
                            (0,11), (1,11), (2,11), # barrel-REC
                            (4,5), (5,6), (6,7), # LEC-LEC
                            (7,8), (8,9), (9,10), 
                            (11,12), (12,13), (13,14), # REC-REC
                            (14,15), (15,16), (16,17)])
                                 
    # load the data
    evtid = int(prefix[-9:])
    logging.info('Event %i, loading data' % evtid)
    hits, particles, truth = trackml.dataset.load_event(
        prefix, parts=['hits', 'particles', 'truth'])

    # apply hit selection
    logging.info('Event %i, selecting hits' % evtid)
    hits, particles = select_hits(hits, truth, particles, pt_min, endcaps, 
                                  remove_noise, remove_duplicates)
    hits = hits.assign(evtid=evtid)
    
    # get truth information for each particle
    hits_by_particle = hits.groupby('particle_id')
    particle_properties = get_particle_properties(hits_by_particle,
                                                  set(layer_pairs), debug=False)
    hits = hits[['hit_id', 'r', 'phi', 'eta', 'z', 'evtid',
                 'layer', 'module_id', 'particle_id']]
    
    # divide detector into sectors
    phi_edges = np.linspace(*phi_range, num=n_phi_sectors+1)
    eta_edges = np.linspace(*eta_range, num=n_eta_sectors+1)
    hits_sectors, sector_info = split_detector_sectors(hits, phi_edges, eta_edges,
                                                       phi_overlap=phi_overlap,
                                                       eta_overlap=eta_overlap)
    
    # calculate particle truth in each sector
    n_track_segs_per_s = {}
    for s, hits_sector in hits_sectors.items():
        hits_sector_by_particle = hits_sector.groupby('particle_id')
        n_track_segs_s = get_n_track_segs(hits_sector_by_particle, set(layer_pairs))
        n_track_segs_per_s[s] = n_track_segs_s
    particle_properties['n_track_segs_per_s'] = n_track_segs_per_s
    
    # graph features and scale
    feature_names = ['r', 'phi', 'z']
    feature_scale = np.array([1000., np.pi / n_phi_sectors, 1000.])

    # Construct the graph
    logging.info('Event %i, constructing graphs' % evtid)
    sectors = [construct_graph(sector_hits, layer_pairs=layer_pairs,
                               phi_slope_max=phi_slope_max, z0_max=z0_max,
                               s=s, feature_names=feature_names,
                               feature_scale=feature_scale,
                               evtid=evtid, module_map=module_map,
                               use_module_map=use_module_map)
               for s, sector_hits in hits_sectors.items()]

    logging.info('Event %i, calculating graph summary' % evtid)
    summary_stats = graph_summary(evtid, sectors, particle_properties,
                                  sector_info, print_per_layer=False)
    
    # Write these graphs to the output directory
    base_prefix = os.path.basename(prefix)
    filenames = [os.path.join(output_dir, f'{base_prefix}_s{i}')
                 for i in range(len(sectors))]
    #except Exception as e:
    #    logging.info(e)
    
    logging.info('Event %i, writing graphs', evtid)    
    for sector, filename in zip(sectors, filenames):
        np.savez(filename, ** dict(x=sector['x'], y=sector['y'], 
                                   particle_id=sector['particle_id'],
                                   edge_attr=sector['edge_attr'],
                                   edge_index=sector['edge_index']))
        
    return {'sectors': sectors, 
            'particle_properties': particle_properties,
            'summary_stats': summary_stats}

def main():
    """Main function"""

    # parse the command line
    args = parse_args()

    # setup logging
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)
    logging.info('Initializing')
    
    # load configuration
    with open(args.config) as f:
        config = yaml.load(f, yaml.FullLoader)
    if args.task==0:
        logging.info(f'Configuration: {config}')
    
    # grab relevant pt_min label
    pt_map = {0: '0p0', 0.5: '0p5', 0.6: '0p6', 0.7: '0p7',
              0.8: '0p8', 0.9: '0p9', 1: '1', 1.1: '1p1',
              1.2: '1p2', 1.3: '1p3', 1.4: '1p4', 1.5: '1p5',
              1.6: '1p6', 1.7: '1p7', 1.8: '1p8', 1.9: '1p9',
              2.0: '2'}
    pt_str = pt_map[config['selection']['pt_min']]

    # find the input files
    input_dir = config['input_dir']
    all_files = os.listdir(input_dir)
    suffix = '-hits.csv'
    #file_prefixes = sorted(os.path.join(input_dir, f.replace(suffix, ''))
    #                       for f in all_files if f.endswith(suffix))
    #file_prefixes = file_prefixes[:config['n_files']]
    file_prefixes = sorted(os.path.join(input_dir, f.replace(suffix, ''))
                           for f in all_files if f.endswith(suffix))

    # restrict files to a given input range
    evtids = [int(prefix[-9:]) for prefix in file_prefixes]
    evtid_max, evtid_min = config['evtid_max'], config['evtid_min']
    if (evtid_min < np.min(evtids)): evtid_min = np.min(evtids)
    if (evtid_max > np.max(evtids)): evtid_max = np.max(evtids)
    file_prefixes = [prefix for prefix in file_prefixes
                     if ((int(prefix.split("00000")[1]) >= evtid_min) and
                         (int(prefix.split("00000")[1]) <= evtid_max))]
    
    # split the input files by number of tasks and select my chunk only
    file_prefixes = np.array_split(file_prefixes, args.n_tasks)[args.task]
    
    # optionally load module maps
    if (config['selection']['use_module_map']):
        module_map_dir = config['module_map_dir']
        module_map = np.load(f"{module_map_dir}/module_map_2_{pt_str}GeV.npy", 
                              allow_pickle=True).item()
        module_map = {key: item.astype(bool) for key, item in module_maps.items()}
    else: module_map = []

    # prepare output
    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'Writing outputs to {output_dir}')

    # process input files with a worker pool
    with mp.Pool(processes=args.n_workers) as pool:
        process_func = partial(process_event, output_dir=output_dir,
                               phi_range=(-np.pi, np.pi), 
                               module_map=module_map,
                               **config['selection'])
        output = pool.map(process_func, file_prefixes)
    
    # analyze output statistics
    logging.info('All done!')
    graph_sectors = [graph['sectors'] for graph in output]
    summary_stats = [graph['summary_stats'] for graph in output]
    n_nodes = np.array([graph_stats['n_nodes'] 
                        for graph_stats in summary_stats])
    n_edges = np.array([graph_stats['n_edges'] 
                        for graph_stats in summary_stats])
    purity = np.array([graph_stats['purity'] 
                       for graph_stats in summary_stats])
    efficiency = np.array([graph_stats['efficiency'] 
                           for graph_stats in summary_stats])
    boundary_fraction = np.array([graph_stats['boundary_fraction'] 
                                  for graph_stats in summary_stats])
    logging.info(f'Events ({evtid_min},{evtid_max}), average stats:\n' +
                 f'...n_nodes: {n_nodes.mean():.0f}+/-{n_nodes.std():.0f}\n' +
                 f'...n_edges: {n_edges.mean():.0f}+/-{n_edges.std():.0f}\n' + 
                 f'...purity: {purity.mean():.5f}+/-{purity.std():.5f}\n' + 
                 f'...efficiency: {efficiency.mean():.5f}+/-{efficiency.std():.5f}\n' + 
                 f'...boundary fraction: {boundary_fraction.mean():.5f}+/-{boundary_fraction.std():.5f}')
    
    # analyze per-sector statistics
    sector_stats_list = [graph_stats['sector_stats'] 
                         for graph_stats in summary_stats]
    neta_sectors = config['selection']['n_eta_sectors']
    nphi_sectors = config['selection']['n_phi_sectors']
    num_sectors = neta_sectors*nphi_sectors
    eta_range_per_s = {s: [] for s in range(num_sectors)}
    phi_range_per_s = {s: [] for s in range(num_sectors)}
    n_nodes_per_s = {s: [] for s in range(num_sectors)}
    n_edges_per_s = {s: [] for s in range(num_sectors)}
    purity_per_s = {s: [] for s in range(num_sectors)}
    efficiency_per_s = {s: [] for s in range(num_sectors)}
    for sector_stats in sector_stats_list:
        for s, stats in sector_stats.items():
            eta_range_per_s[s] = stats['eta_range']
            phi_range_per_s[s] = stats['phi_range']
            n_nodes_per_s[s].append(stats['n_nodes'])
            n_edges_per_s[s].append(stats['n_edges'])
            purity_per_s[s].append(stats['purity'])
            efficiency_per_s[s].append(stats['efficiency'])
            
    for s in range(num_sectors):
        eta_range_s = eta_range_per_s[s]
        phi_range_s = phi_range_per_s[s]
        n_nodes_s = np.array(n_nodes_per_s[s])
        n_edges_s = np.array(n_edges_per_s[s])
        purity_s = np.array(purity_per_s[s])
        efficiency_s = np.array(efficiency_per_s[s])
        logging.info(f'Event ({evtid_min},{evtid_max}), Sector {s}, average stats:\n' +
                     f'...eta_range: ({eta_range_s[0]:.3f},{eta_range_s[1]:.3f})\n' + 
                     f'...phi_range: ({phi_range_s[0]:.3f},{phi_range_s[1]:.3f})\n' + 
                     f'...n_nodes: {n_nodes_s.mean():.0f}+/-{n_nodes_s.std():.0f}\n' +
                     f'...n_edges: {n_edges_s.mean():.0f}+/-{n_edges_s.std():.0f}\n' + 
                     f'...purity: {purity_s.mean():.5f}+/-{purity_s.std():.5f}\n' + 
                     f'...efficiency: {efficiency_s.mean():.5f}+/-{efficiency_s.std():.5f}')
        
if __name__ == '__main__':
    main()
