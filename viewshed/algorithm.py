import numpy as np
from scipy.interpolate import RegularGridInterpolator as Interpolator, interp1d
from enum import IntEnum
from typing import Dict
from math import floor


class Crossing(IntEnum):
    X_ONLY = 1
    Y_ONLY = 2


class BaseAlgorithm():
    """原则：
    1. 算法中所有点的坐标都采用 DEM 矩阵坐标
    2. DEM 矩阵的 [0, 0] 对应最左上角像元的中点，即 ( ymax - yunit / 2, xmin + xunit / 2 )
                                              (      i          ,        j         )
    3. 可以使用 i2f 和 f2i 相互转换，注意 f2i 只能给出点所在的具体索引。
    """

    def __init__(self, transform: Dict, dem: np.ndarray) -> None:
        self.xmin = transform[0]
        self.ymax = transform[3]
        self.xmax = self.xmin + dem.shape[1] * transform[1]
        self.ymin = self.ymax + dem.shape[0] * transform[5]

        self.xunit = transform[1]
        self.yunit = -transform[5]

        self.dem = dem  # 注意 [0, 0] 位置对应的经纬度应该是 (ymax, xmin)
        y = np.arange(dem.shape[1])
        x = np.arange(dem.shape[0])
        self.interp_dem = Interpolator((y, x), dem)

    def i2f(self, i: int, j: int):
        # 给定栅格索引，返回栅格单元的中心点的经纬度
        return self.ymax - (i + 0.5) * self.yunit, self.xmin + (j + 0.5) * self.xunit

    def f2i(self, y: float, x: float):
        # 给定某经纬度，返回所属于的栅格单元索引
        return floor((self.ymax - y) / self.yunit), floor((x - self.xmin) / self.xunit)

    def crossing_points(self, startf, endi, crossing=Crossing.X_ONLY) -> np.ndarray:
        # 根据选择的交叉类型对所有数据进行差值
        starti = self.f2i(*startf)
        if crossing == Crossing.X_ONLY:
            i = np.arange(starti[0], endi[0] + 1)
            f = interp1d([starti[0], endi[0]], [starti[1], endi[1]])
            j = f(i)
        else:
            j = np.arange(starti[1], endi[1] + 1)
            f = interp1d([starti[1], endi[1]], [starti[0], endi[0]])
            i = f(j)
        return i, j  # 获得每个交点坐标

    def interpolate_los(
        self,
        i: np.ndarray,
        j: np.ndarray,
        start_ele: float,
        end_ele: float,
        crossing=Crossing.X_ONLY
    ) -> np.ndarray:
        # 求 LOS 线在每个整点处的差值高程
        if crossing == Crossing.X_ONLY:
            locs = [i[0], i[-1]]
            indice = i
        else:
            locs = [j[0], j[-1]]
            indice = j

        elevs = [start_ele, end_ele]
        f = interp1d(locs, elevs)

        # interpolate from starti to endi
        return f(indice)

    def interpolate_elev(self, i, j):
        # 求 LOS 线在每个整点处的 DEM 高程差值
        return self.interp_dem(np.dstack((i, j)))[0]


if __name__ == "__main__":
    from osgeo import gdal
    import os

    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    ds = gdal.Open('./tests/data/dem500.tif')
    band_data = ds.GetRasterBand(1).ReadAsArray()
    geoTransform = ds.GetGeoTransform()

    # draw band_data
    import matplotlib.pyplot as plt
    plt.imshow(band_data[:500, :500])

    # 以下代码演示对于一个视点、一个边界点如何判断是否可视
    # 真正的算法应当考虑将下面逻辑向量化，即：
    # 1. 初始化**一系列**边界点，如 R2 算法初始化所有边界点
    # 2. 将下面逻辑抽象成为一个 np.vectorize 的函数，将所有边界点输入
    # 3. 批量拿到所有交点的 LOS 值和实际 DEM 差值
    # 4. 对于内部的点，找最近的（LOS, DEM）数值对并判断之

    # 初始化算法
    # 此处会定义算法运算范围、定义差值函数等，只需运行一次
    alg = BaseAlgorithm(geoTransform, band_data)

    # 假设视点及视点处观察者高度：
    # 注意视点坐标是从 point layer 上面拿来的，属于地理坐标系的距离（测试数据是 [经,纬]），float 类型
    # 另外注意矩阵索引第一维对应 经度（y轴）+ 逆向，第二维对应 纬度（x轴）+ 正向。
    start, start_elev = (36, 76.3), 5000

    # 假设想要求的边界某点坐标
    # 注意边界点坐标是矩阵的索引，因为是迭代边界拿到的。
    # 此处测试用 DEM 矩阵尺寸 500 x 500，边界点选择 （100，499）
    # 实际当中应该生成 8 个边界点序列，然后一次性输入
    endi = (100, 499)

    # 这里还有一个可视性 == True 的样例替换上方数据
    # tips: alg.f2i 函数可以将经纬度和 DEM 矩阵索引进行转换
    # start, start_elev = (35.4708,76.5656), 6447
    # endi = alg.f2i(35.4550,76.5472)

    # 设定相交于横轴还是纵轴
    # 或许 8 个区域应该交叉用 X_ONLY 和 Y_ONLY 确保精度？
    # 即：12点钟方向和 3 点钟方向之间，前半部分用 X_ONLY, 后半部分用 Y_ONLY
    cross = Crossing.X_ONLY

    # 首先求视点和边界点连线与各个像元的交点
    # crossing 表示交点出现在 x 轴上还是 y 轴上
    # 返回一串点集的索引坐标差值，保证 i j 范围永远在矩阵的索引之内。
    i, j = alg.crossing_points(start, endi, cross)
    # 统计交点数目
    shp = np.dstack((i, j))[0].shape
    print('共有 {} 个交点'.format(shp[0]))

    # 求视点和边界点连线上的所有交点的高程差值
    interpolated_elev = alg.interpolate_elev(i, j)
    assert alg.dem[int(i[0]), int(j[0])] == interpolated_elev[0] and alg.dem[int(
        i[-1]), int(j[-1])] == interpolated_elev[-1]

    # 求视点和边界点 LOS 上所有交点的高程差值
    # 应该是三角形的斜边那条线
    interpolated_los = alg.interpolate_los(
        i, j, start_elev, alg.dem[endi[0], endi[1]], cross)
    assert start_elev == interpolated_los[0]
    assert alg.dem[endi[0], endi[1]] == interpolated_los[-1]

    # 两者差值交点个数是否一致
    assert interpolated_elev.shape == interpolated_los.shape

    visible = np.all(interpolated_los >= interpolated_elev)

    print("可见性：", visible)
