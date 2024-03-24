import codecs

import open3d as o3d
import numpy as np

import skimage.measure

import tool
from logger import log


def read_txt_point_cloud(file_path,delimiter=";",data=None):
    with open(file_path, "r") as f:
        lines = f.readlines()
        if len(lines)==0:
            return None,None
        xyz = []
        rgb = []
        for line in lines:
            values = line.strip().split(delimiter)

            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
            xyz.append([x, y, z])
            rgb.append([int(v) for v in values[3:]])

            if data is not None:
                data[round(x/2),round(y/2),round(z/2)]=int(values[4])

    xyz_array = np.array(xyz)

    rgb_array = np.array(rgb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_array)
    pcd.colors = o3d.utility.Vector3dVector(rgb_array / 255.0)
    return pcd,xyz_array



def writeTxtFile(text,fileName):

    file_object = codecs.open(fileName,'w','utf-8')
    file_object.write(text)
    file_object.close( )


def readFile(fileName):
    with  codecs.open(fileName, "r", "utf-8") as f:
        lines = f.readlines()
        lines2 = []
        for line in lines:
            lines2.append(line)

    return lines2


def mesh_rebuild(pointCloudFilePath, meshFilePath, color=[1,1,1], deal4PointCloud=False):
    meshFilePathGbk=meshFilePath.encode("gbk")

    pcd,xyz_array = read_txt_point_cloud(pointCloudFilePath)
    if pcd is None:
        log.info(" empty, ignore to rebuild mesh. pointCloudFilePath={}".format(pointCloudFilePath))
        return
    log.info("%s pcd.get_min_bound()=%s"%(pointCloudFilePath,pcd.get_min_bound()))
    log.info("pcd.get_max_bound()=%s"%pcd.get_max_bound())
    max_index = round(max(pcd.get_max_bound())/2)+2
    log.info("max_index={}".format(max_index))

    data = np.zeros((max_index, max_index, max_index), dtype=np.uint8)

    if deal4PointCloud:
        pcd.points = o3d.utility.Vector3dVector(xyz_array/2)
        log.info("meshFilePath ={}".format(meshFilePath))
        o3d.io.write_point_cloud(meshFilePathGbk,pcd)
        return

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    colorMin=None
    colorMax=None
    log.info("pcd.points count=%s"%len(pcd.points))
    for i,point in enumerate(pcd.points):
        color_v = pcd.colors[i]
        v=max(color_v)
        data[round(point[0]/2),round(point[1]/2),round(point[2]/2)]=v*255
        if colorMin is None or v<colorMin:
            colorMin=v
        if colorMax is None or v>colorMax:
            colorMax=v
    log.info("min max=%s - %s"%(colorMin,colorMax))
    if colorMin==0 and colorMax==0:
        return

    verts, faces, _, _ = skimage.measure.marching_cubes(data, level=1,step_size=1)


    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    mesh = mesh.filter_smooth_simple(number_of_iterations=5)

    log.info("save mesh file to %s"%meshFilePath)
    tool.create_parent_dir(meshFilePath)

    o3d.io.write_triangle_mesh(meshFilePathGbk, mesh, write_ascii=True)

    alpha=0.5
    alpha2=round(alpha*255)
    colors = np.asarray(mesh.vertex_colors)
    lines = readFile(meshFilePath)

    result_lines=[]
    for line in lines:
        line=line.strip()
        if len(line.split(" "))==9:
            line="%s %s"%(line, alpha2)
        result_lines.append(line)
        if line.strip().startswith("property uchar blue"):
            result_lines.append("property uchar alpha")

    writeTxtFile("\n".join(result_lines),meshFilePath)
    



