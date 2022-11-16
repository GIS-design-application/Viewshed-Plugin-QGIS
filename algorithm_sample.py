import numpy as np
def viewshed_demo(px:float, py:float, pz:float,
					dem_array:np.ndarray, dem_config:dict, max_distance=10000):
	"""一个viewshed算法的demo,主要是用来规范接口和引用方式

	Args:
		px (float): 视点的x坐标
		py (float): 视点的y坐标
		pz (float): 视点的z坐标（高程）
		dem_array (np.ndarray): DEM数据
		dem_config (dict): DEM的配置信息,包括geotransform和projection
			geotransform (tuple): gdal geotransform
			projection (str): gdal projection in wkt
		max_distance (int, optional): 最大可视距离 Defaults to 10000. 
		#? max_distance 有用吗? 这个参数是Copilot推荐的我就加上了

	Returns:
		np.ndarray: 返回一个与dem_array同样大小的数组,可视区域为1,不可视区域为0
	"""

	return -dem_array