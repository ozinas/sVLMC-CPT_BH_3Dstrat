# -*- coding: utf-8 -*-
"""
Created on Mon May 22 10:39:19 2023

@author: ozinas
"""

import numpy as np

import pandas as pd
from glob import glob
import gpytorch 
import torch
from sklearn.preprocessing import LabelEncoder

def load_data(file):
    return pd.read_excel(glob("..\\Input\\"+file)[0]).to_numpy()

def load_data_df(file):
    return pd.read_excel(glob("..\\Input\\"+file)[0])

def load_locs(file):
    return pd.read_excel(glob("..\\Input\\"+file)[0], usecols=["X", "Y", "Z"]).to_numpy()

def load_coords(file):
    return pd.read_excel(glob("..\\Input\\"+file)[0], usecols=[0, 1, 2, 3]).to_numpy()

def load_bh_test_coords(file):
    return pd.read_excel(glob("..\\Input\\"+file)[0], skiprows = [0], usecols=[1, 2]).to_numpy()

def load_bounds(file):
    return pd.read_excel(glob("..\\Input\\"+file)[0]).to_numpy()

def data_processing_with_BH_scaled(num_cpts, num_bhs, dis_depth, 
                             num_vertical_des_points,
                             cpt_par):
    
    train_x = load_locs("training_locations.xlsx")
    
    if cpt_par == "qc":
        cpt_par_data_full = load_data("training_data_qc.xlsx")
    elif cpt_par == "fs":
        cpt_par_data_full = load_data("training_data_fs.xlsx")
    elif cpt_par == "Ic":
        cpt_par_data_full = load_data("training_data_Ic.xlsx")
    
    bh_data = cpt_par_data_full[:, -num_bhs:]
    
    label_encoder = LabelEncoder()
    # Fit the LabelEncoder to the USCS class labels and transform them into categorical variables
    encoded_classes = label_encoder.fit_transform(bh_data.ravel(order="f")).reshape(bh_data.shape, order = "f")
    mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    
    encoded_classes = np.array(encoded_classes, dtype = float)
    encoded_classes[encoded_classes == np.max(encoded_classes)] = np.nan
    cpt_par_data_full[:, -num_bhs:] = encoded_classes
    cpt_par_data_full = np.array(cpt_par_data_full, dtype=float)

    coords = load_coords("CPT_BH_coordinates.xlsx")


    N, M = cpt_par_data_full.shape
    z = np.round(train_x[:,2], 2)


    Ic_data = cpt_par_data_full[dis_depth:, :]
    # =============================================================================
    z_mat = z.reshape(N  , M , order = "F")
    # =============================================================================
    coords = coords[coords[:, 1].argsort()]
    

    xy_sorted = coords[:, [1,2]]  - coords[0, [1, 2]] 
    x_sorted = xy_sorted[:, 0]
    y_sorted = xy_sorted[:, 1]

    Dmax = max(np.max(x_sorted), np.max(y_sorted))

    x_sc  = x_sorted / Dmax
    y_sc  = y_sorted / Dmax

    coords_py = coords[:,0] - 1
    indices_ex = np.array(coords_py[1:], dtype = int)

    # xy_sorted = coords[:, [1,2]]  - coords[0, [1,2]] 
    x_sorted = x_sc[1:]
    y_sorted = y_sc[1:]

    x_sorted_mat = np.repeat(x_sorted, N-dis_depth).reshape(N-dis_depth , 
                                                            M , order = "F")
    y_sorted_mat = np.repeat(y_sorted, N-dis_depth).reshape(N-dis_depth ,
                                                            M , order = "F")
    z_sorted_mat = z_mat[dis_depth:, :]

    for i, val in enumerate(indices_ex):
        if val > 5:
            indices_ex[i] = indices_ex[i] - 1


    Ic_data_re = Ic_data[:, indices_ex]
    N, M = Ic_data_re.shape
    
    
    col_counts = np.zeros((Ic_data_re.shape[1], 2))
    for i, col in enumerate(Ic_data_re.T):
        col_counts[i, 1] = np.count_nonzero(~np.isnan(col))
        col_counts[i, 0] = i

    col_counts_sorted = col_counts[col_counts[:, 1].argsort()][::-1]
    idx_cols = np.array(np.sort(col_counts_sorted[: num_cpts, 0]), dtype = int)    
    data, x_sc, y_sc, z_sc, xt, yt, zt, idx   = data_select_with_BH(Ic_data_re,
                                                            x_sorted_mat, y_sorted_mat, 
                                                            z_sorted_mat, idx_cols, num_vertical_des_points, 
                                                            N, M, num_cpts)
    for i, dat in enumerate(data.T):   
        for j in range(len(dat)-1):
            if np.isnan(dat[j]) and ~np.isnan(dat[j-1]) and ~np.isnan(dat[j+1]):
                data[j, i] = (dat[j-1] + dat[j+1]) / 2
            elif np.isnan(dat[j]) and np.isnan(dat[j-1]) and ~np.isnan(dat[j+1]) and ~np.isnan(dat[j-2]):
                data[j, i] = (dat[j-2] + dat[j+1]) / 2
                data[j-1, i] = (dat[j-2] + dat[j+1]) / 2       
                
    grid = []
    grid.append(torch.from_numpy(zt).float())
    grid.append(torch.from_numpy(xt).float())
    grid.append(torch.from_numpy(yt).float())
    
    Dxy = np.max(xy_sorted)
    Dz = np.max(z_mat)
    depth_0 = z_mat[dis_depth, 0]
    
    return data, xt, yt, zt, grid, z_mat, mapping,\
        Dxy, Dz, depth_0
        
def data_select_with_BH(data, xcoord, ycoord, zcoord, idx_cols, numElems_z, Nf, Mf, num_cpts):
    
    N, M = data.shape
    numElems_x = len(idx_cols)
    
    
    data_log = data
    v = np.linspace(0, N, numElems_z, endpoint=False)

    v += 1 / N  
    idz = np.round(v).astype(int)
    data = data_log[np.newaxis, idz].squeeze()
    
    y = np.linspace(0, data.shape[1], numElems_x, endpoint=False)
    y += 1 / M
    idx = np.round(y).astype(int)
    
    idx = idx_cols
    data = data[:, idx]

    z_sc = (gpytorch.utils.grid.scale_to_bounds(zcoord, 0, 1) / 0.95)
        
    
    x_p = xcoord.reshape(Nf  , M , order = "F")[idz, :][:, idx]
    y_p = ycoord.reshape(Nf  , M , order = "F")[idz, :][:, idx]
    z_p = z_sc.reshape(Nf , M , order = "F")[idz, :][:, idx]
    
    return data, x_p, y_p, z_p, x_p[0, :], y_p[0, :], z_p[:, 0] , idx
