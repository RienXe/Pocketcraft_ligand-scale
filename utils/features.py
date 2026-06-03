import torch
import torch.nn.functional as F

def get_raw_atom_feature(res_name, atom_name, element, aa_feat_dict, atom_feat_dict):
    """Priority: dict_aa_feature > dict_atom_feature"""
    if res_name in aa_feat_dict and atom_name in aa_feat_dict[res_name]:
        return aa_feat_dict[res_name][atom_name]
    return atom_feat_dict.get(element, 0)

def get_weighted_features(surface_pts, atom_pts, protein_atoms, feat_config, k=3):
    aa_feat_dict = feat_config['dict_aa_feature']
    atom_feat_dict = feat_config['dict_atom_feature']
    regular_aa = feat_config['regular_aa']
    
    atom_feature_list = []
    for a in protein_atoms:
        val = get_raw_atom_feature(a.res_name, a.atom_name, a.element, aa_feat_dict, atom_feat_dict)
        aa_idx = regular_aa.index(a.res_name) if a.res_name in regular_aa else regular_aa.index("UNK")
        
        val_oh = F.one_hot(torch.tensor(val), num_classes=11)
        aa_oh = F.one_hot(torch.tensor(aa_idx), num_classes=len(regular_aa))
        atom_feature_list.append(torch.cat([val_oh, aa_oh], dim=-1).float())
    
    atom_feats = torch.stack(atom_feature_list)
    dist = torch.cdist(surface_pts, atom_pts)
    top_dist, indices = torch.topk(dist, k, largest=False)
    weights = 1.0 / (top_dist + 1e-6)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    return (atom_feats[indices] * weights.unsqueeze(-1)).sum(dim=1)
