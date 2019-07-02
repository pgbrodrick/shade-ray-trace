

import argparse
import gdal
import math
import numpy as np
from tqdm import tqdm

import bresenham_line



parser = argparse.ArgumentParser('Calculate shade mask using a ray trace from OBS and TCH model')
parser.add_argument('obs_file', type=str)
parser.add_argument('igm_file', type=str)
parser.add_argument('dsm_file', type=str)
parser.add_argument('out_file', type=str)
parser.add_argument('-solar_azimuth_b', type=int, default=4)
parser.add_argument('-solar_zenith_b', type=int, default=5)
parser.add_argument('-of_type', type=str, default='GTiff')

args = parser.parse_args()







obs_set = gdal.Open(args.obs_file, gdal.GA_ReadOnly)
igm_set = gdal.Open(args.igm_file, gdal.GA_ReadOnly)
dsm_set = gdal.Open(args.dsm_file, gdal.GA_ReadOnly)

assert obs_set is not None, 'Cannot open {}'.format(args.obs_file)
assert igm_set is not None, 'Cannot open {}'.format(args.igm_file)
assert dsm_set is not None, 'Cannot open {}'.format(args.dsm_file)

driver = gdal.GetDriverByName(args.of_type)
driver.Register()
outDataset = driver.Create(args.out_file,obs_set.RasterXSize,obs_set.RasterYSize,1,gdal.GDT_Byte)
outDataset.SetGeoTransform(obs_set.GetGeoTransform())
outDataset.SetProjection(obs_set.GetProjection())

dsm = dsm_set.ReadAsArray()
dsm_trans = dsm_set.GetGeoTransform()
dsm_max_px = int(math.ceil(math.sqrt(dsm_set.RasterXSize**2 + dsm_set.RasterYSize**2)))

for _line in tqdm(range(obs_set.RasterYSize),ncols=80):

    obs_line = np.squeeze(obs_set.ReadAsArray(0,_line,obs_set.RasterXSize,1))
    igm_line = np.squeeze(igm_set.ReadAsArray(0,_line,igm_set.RasterXSize,1))

    solar_azimuth = np.squeeze(obs_line[args.solar_azimuth_b-1,:])
    solar_zenith = np.squeeze(obs_line[args.solar_zenith_b-1,:])


    # calculate the pre-rounded version (want for edge calc...will round below)
    dsm_target_px_x = (igm_line[0,:] - dsm_trans[0])/dsm_trans[1]
    dsm_target_px_y = (igm_line[1,:] - dsm_trans[3])/dsm_trans[5]

    # Find edge pixels
    dsm_edge_px_x = dsm_target_px_x + dsm_max_px * np.sin(np.pi / 180. * solar_azimuth)
    dsm_edge_px_y = dsm_target_px_y + dsm_max_px * np.cos(np.pi / 180. * solar_azimuth)


    shp = dsm_target_px_x.shape
    dsm_edge_px_x = np.round(np.minimum(np.maximum(dsm_edge_px_x, np.zeros(shp)), np.ones(shp) * dsm_set.RasterXSize - 1)).astype(int)
    dsm_edge_px_x = np.round(np.minimum(np.maximum(dsm_edge_px_x, np.zeros(shp)), np.ones(shp) * dsm_set.RasterXSize - 1)).astype(int)
    dsm_edge_px_y = np.round(np.minimum(np.maximum(dsm_edge_px_y, np.zeros(shp)), np.ones(shp) * dsm_set.RasterYSize - 1)).astype(int)
    dsm_edge_px_y = np.round(np.minimum(np.maximum(dsm_edge_px_y, np.zeros(shp)), np.ones(shp) * dsm_set.RasterYSize - 1)).astype(int)

    dsm_target_px_x = np.round(dsm_target_px_x).astype(int)
    dsm_target_px_y = np.round(dsm_target_px_y).astype(int)

    dsm_target_px = np.transpose(np.vstack([dsm_target_px_x,dsm_target_px_y]))
    dsm_edge_px = np.transpose(np.vstack([dsm_edge_px_x,dsm_edge_px_y]))

    del dsm_target_px_x, dsm_target_px_y, dsm_edge_px_x, dsm_edge_px_y


    sunlit_line = np.zeros(obs_set.RasterXSize)
    for _px in range(len(solar_azimuth)):
        line_px = bresenham_line.bresenhamline(dsm_target_px[_px,:].reshape(1,-1), dsm_edge_px[_px,:].reshape(1,-1), max_iter=-1)
        surf = dsm[line_px[:,1],line_px[:,0]]

        sunlit_line[_px] = int(dsm[dsm_target_px[_px,1],dsm_target_px[_px,0]] > np.nanmax(surf))


    outDataset.GetRasterBand(1).WriteArray(sunlit_line.reshape(1,-1),0,_line)
    outDataset.FlushCache()










