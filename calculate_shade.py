 

import argparse
from osgeo import gdal
import math
import numpy as np
from tqdm import tqdm

import bresenham_line
'''
NOTES
OBS='/Users/carissaderanek/Library/CloudStorage/Box-Box/ray_trace/flightline_input/OBS_Data_negnan.tif'
IGM='/Users/carissaderanek/Library/CloudStorage/Box-Box/ray_trace/flightline_input/IGM_Data_negnan_filledcoords.tif'
DSM='/Users/carissaderanek/Library/CloudStorage/Box-Box/ray_trace/flightline_input/dsm_negnan.tif'
OUT='/Users/carissaderanek/Library/CloudStorage/Box-Box/ray_trace/flightline_input/filledcoordsout.tif'
python calculate_shade.py $OBS $IGM -dsm_file $DSM $OUT
How I created the filled coords:
>>> igm = rioxarray.open_rasterio(path_to_igm_original_tif)
>>> new_igm_x=np.array([igm.x.values for i in range(igm.values[0].shape[0])])
>>> new_igm_y=np.array([igm.y.values for i in range(igm.values[0].shape[1])]).T
>>> new_igm = igm.copy(data=[new_igm_x,new_igm_y,igm.values[2]])
>>> new_igm.rio.to_raster('/Users/carissaderanek/Library/CloudStorage/Box-Box/ray_trace/flightline_input/IGM_Data_negnan_filledcoords.tif')
Only edit to original code (on line 169):
sunlit_line = np.where(igm_line[2, :] == -9999, -9999, sunlit_line) # at very end, mask values where there ws no IGM data originally
'''

def main():
    parser = argparse.ArgumentParser(
        'Calculate shade mask using a ray trace from OBS and surface model')
    parser.add_argument('obs_file', type=str)
    parser.add_argument('igm_file', type=str)
    parser.add_argument('-dsm_file', type=str)
    parser.add_argument('-dem_file', type=str)
    parser.add_argument('-tch_file', type=str)

    parser.add_argument('out_file', type=str)
    parser.add_argument('-solar_azimuth_b', type=int, default=4)
    parser.add_argument('-solar_zenith_b', type=int, default=5)
    parser.add_argument('-of_type', type=str, default='GTiff')

    args = parser.parse_args()

    use_dsm = args.dsm_file is not None
    use_tch = (args.dem_file is not None and args.tch_file is not None)

    if use_dsm is False and use_tch is False:
        print('Either dsm_file or tch_file and dem_file must be specified')

    if use_dsm and use_tch:
        print('Specify only dsm_file or tch_file and dem_file')

    obs_set = gdal.Open(args.obs_file, gdal.GA_ReadOnly)
    igm_set = gdal.Open(args.igm_file, gdal.GA_ReadOnly)
    if use_dsm:
        dsm_set = gdal.Open(args.dsm_file, gdal.GA_ReadOnly)
    else:
        dem_set = gdal.Open(args.dem_file, gdal.GA_ReadOnly)
        tch_set = gdal.Open(args.tch_file, gdal.GA_ReadOnly)

    assert obs_set is not None, 'Cannot open {}'.format(args.obs_file)
    assert igm_set is not None, 'Cannot open {}'.format(args.igm_file)

    if use_dsm:
        assert dsm_set is not None, 'Cannot open {}'.format(args.dsm_file)
    else:
        assert dem_set is not None, 'Cannot open {}'.format(args.dem_file)
        assert tch_set is not None, 'Cannot open {}'.format(args.tch_file)

    driver = gdal.GetDriverByName(args.of_type)
    driver.Register()
    outDataset = driver.Create(args.out_file, obs_set.RasterXSize,
                               obs_set.RasterYSize, 1, gdal.GDT_Byte)
    outDataset.SetGeoTransform(obs_set.GetGeoTransform())
    outDataset.SetProjection(obs_set.GetProjection())

    if use_dsm:
        surface = dsm_set.ReadAsArray()
        dsm_trans = list(dsm_set.GetGeoTransform())
        x_size = dsm_set.RasterXSize
        y_size = dsm_set.RasterYSize
    else:
        surface = dem_set.ReadAsArray() + tch_set.ReadAsArray()
        dsm_trans = list(dem_set.GetGeoTransform())
        x_size = dem_set.RasterXSize
        y_size = dem_set.RasterYSize
    dsm_trans[0] += float(dsm_trans[1]) / 2.
    dsm_trans[3] += float(dsm_trans[5]) / 2.


    # outDataset.GetRasterBand(1).WriteArray(np.zeros((igm_set.RasterYSize,igm_set.RasterXSize)),0,0)
    for _line in tqdm(range(obs_set.RasterYSize), ncols=80):

        obs_line = np.squeeze(obs_set.ReadAsArray(0, _line, obs_set.RasterXSize, 1))
        igm_line = np.squeeze(igm_set.ReadAsArray(0, _line, igm_set.RasterXSize, 1))

        solar_azimuth = np.squeeze(obs_line[args.solar_azimuth_b-1, :])
        solar_zenith = np.squeeze(obs_line[args.solar_zenith_b-1, :])
        # solar_azimuth[solar_azimuth < 0] = 0
        # print(solar_azimuth)

        # adjust for sun width
        #solar_zenith = solar_zenith - 0.53
        # solar_zenith[solar_zenith < 0] = 0

        # calculate the pre-rounded version (want for edge calc...will round below)
        dsm_target_px_x = (igm_line[0, :] - dsm_trans[0])/dsm_trans[1]
        dsm_target_px_y = (igm_line[1, :] - dsm_trans[3])/dsm_trans[5]

        # Find per-pixel distance to nearest edge in direction of sun
        dsm_max_px = np.ones(dsm_target_px_x.shape) * \
            int(math.ceil(math.sqrt(x_size**2 + y_size**2)))

        subset = np.logical_and(solar_azimuth > 0, solar_azimuth <= 180)
        dsm_max_px[subset] = np.minimum(dsm_max_px[subset], (x_size -
                                                             dsm_target_px_x[subset])/np.abs(np.sin(np.pi / 180 * solar_azimuth[subset])))

        subset = np.logical_not(subset)
        dsm_max_px[subset] = np.minimum(
            dsm_max_px[subset], dsm_target_px_x[subset]/np.abs(np.sin(np.pi / 180 * solar_azimuth[subset])))

        subset = np.logical_and(solar_azimuth > 90, solar_azimuth <= 270)
        dsm_max_px[subset] = np.minimum(dsm_max_px[subset], (y_size -
                                                             dsm_target_px_y[subset])/np.abs(np.cos(np.pi / 180 * solar_azimuth[subset])))

        subset = np.logical_not(subset)
        dsm_max_px[subset] = np.minimum(
            dsm_max_px[subset],  dsm_target_px_y[subset]/np.abs(np.cos(np.pi / 180 * solar_azimuth[subset])))

        # Find edge pixels
        dsm_edge_px_x = dsm_target_px_x + dsm_max_px * np.sin(np.pi / 180. * solar_azimuth)
        dsm_edge_px_y = dsm_target_px_y - dsm_max_px * np.cos(np.pi / 180. * solar_azimuth)

        # Round edge pixels
        shp = dsm_target_px_x.shape
        dsm_edge_px_x = np.floor(np.minimum(np.maximum(dsm_edge_px_x, np.zeros(
            shp)), np.ones(shp) * x_size - 1)).astype(int)
        dsm_edge_px_y = np.floor(np.minimum(np.maximum(dsm_edge_px_y, np.zeros(
            shp)), np.ones(shp) * y_size - 1)).astype(int)

        # Round target pixels now that egdes have been found
        dsm_target_px_x = np.floor(dsm_target_px_x).astype(int)
        dsm_target_px_y = np.floor(dsm_target_px_y).astype(int)

        # combine x/y arrays
        dsm_target_px = np.transpose(np.vstack([dsm_target_px_x, dsm_target_px_y]))
        dsm_edge_px = np.transpose(np.vstack([dsm_edge_px_x, dsm_edge_px_y]))

        valid_subset = np.where(np.logical_and.reduce((dsm_target_px_x >= 0, 
                                                       dsm_target_px_y >= 0, 
                                                       dsm_target_px_x < dsm_set.RasterXSize, 
                                                       dsm_target_px_y < dsm_set.RasterYSize)))
        del dsm_target_px_x, dsm_target_px_y, dsm_edge_px_x, dsm_edge_px_y

        sunlit_line = np.zeros(obs_set.RasterXSize)
        for _px in valid_subset[0]:
            # Find line of pixels in direction of sun
            line_px = bresenham_line.bresenhamline(dsm_target_px[_px, :].reshape(
                1, -1), dsm_edge_px[_px, :].reshape(1, -1), max_iter=-1)

            # print(_px, dsm_target_px[_px, :],dsm_edge_px[_px, :],line_px)

            # Find surface value at each point on line

            surf = surface[line_px[:, 1], line_px[:, 0]]
            surf[np.isfinite(surf) == False] = -9999

            # Calculate [horizontal] distance of each point on line from target
            dist = np.sqrt(np.sum(np.power(line_px - dsm_target_px[_px, :], 2), axis=1))

            # Calculate height of solar ray at each point on line
            solar_height = surface[dsm_target_px[_px, 1], dsm_target_px[_px, 0]
                                   ] + dist * np.tan(np.pi / 180 * (90-solar_zenith[_px]))

            # Sunlit if solar height is always higher than relative surface (IE, no ray intersection)
            sunlit_line[_px] = np.all(solar_height > surf)
            sunlit_line = np.where(igm_line[2, :] == -9999, -9999, sunlit_line)

        outDataset.GetRasterBand(1).WriteArray(sunlit_line.reshape(1, -1), 0, _line)
        outDataset.FlushCache()


if __name__ == "__main__":
    main()
