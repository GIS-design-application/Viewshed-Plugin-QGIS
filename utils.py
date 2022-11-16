from osgeo import gdal
import numpy as np

def get_band_data(url: str, band: int = 1):
	"""
	get band data from raster
	"""
	ds = gdal.Open(url)
	band_data = ds.GetRasterBand(band).ReadAsArray()
	return np.array(band_data)

def save_raster(url:str, data: np.array, config: dict):
	"""
	save raster data to file
	"""
	driver_type = judge_driver_type(url)
	driver = gdal.GetDriverByName(driver_type)

	outdata = driver.Create(url, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
	outdata.SetGeoTransform(config['geotransform'])
	outdata.SetProjection(config['projection'])
	outdata.GetRasterBand(1).WriteArray(data)
	outdata.FlushCache()

def judge_driver_type(url):
	"""
	judge GDAL driver type
	"""
	driver_type = url.split('.')[-1]
	if driver_type == 'tif':
		return 'GTiff'
	elif driver_type == 'img':
		return 'HFA'
	elif driver_type == 'jpg':
		return 'JPEG'
	elif driver_type == 'png':
		return 'PNG'
	elif driver_type == 'bmp':
		return 'BMP'
	elif driver_type == 'jp2':
		return 'JP2'
	else:
		raise ValueError('Unsupported file type!')

def get_raster_config(dem):
	"""
	get raster config
	"""
	projection = dem.dataProvider().crs().toWkt()
	# generate gdal geotransform
	extent = dem.extent()
	xmin = extent.xMinimum()
	ymax = extent.yMaximum()
	xres = dem.rasterUnitsPerPixelX()
	yres = dem.rasterUnitsPerPixelY()
	geotransform = (xmin, xres, 0, ymax, 0, -yres)
	return {'geotransform': geotransform, 'projection': projection}