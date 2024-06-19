import numpy as np
from pathlib import Path
import os
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R


"""
Function, which loads a training pointcloud.

Input:
objectType: A string, which specifies the type of training pointcloud (e.g. "book", "cup", ..)
pointcloud_idx: An integer, which specifies, which pointcloud will be loaded.
use_voxel_downsampling: Boolean, if true, voxe_downsampling will be used.
voxel_size: float, parameter used for voxel_downsampling.

Output:
Returns a pointcloud object.
"""
def loadObjectPointcloud(objectType, use_voxel_downsampling=True, voxel_size=0.01):

        
    current_path = Path(__file__).parent
    #Load Pointcloud
    pcd = o3d.io.read_point_cloud(os.path.join(current_path, "scenes", "new", "objects", objectType, objectType+".ply"),
                                    remove_nan_points=True, remove_infinite_points=True)
    #o3d.visualization.draw_geometries([pcd])
    #Remove duplicated points
    pcd = pcd.remove_duplicated_points()
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Down-sample the loaded point cloud to reduce computation time
    if use_voxel_downsampling:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)


    return pcd
    

"""
Function used for projecting the 3D pointcloud onto a 2D scene using a pinhole camera model.

#Input:
    points: np.array where each row represents a point. Each row contains 3 values, corresponding to x, y, z coordinates.
    colors: np.array where each row corresponds to the same row in points. Each row contains 3 values, which represent the color in RGB-convention.
#Output:
    points_projected: np.array where each row represents a point. Each row contains 2 values, corresponding to x, y coordinates. The 3d points have been projected to a 2D scene.
    colors: np.array where each row corresponds to the same row in points_projected. Each row contains 3 values, which represent the color in BGR-convention.

"""


def load_view_point(pcd, para_pose):

    translation = para_pose[:3]
    rx = para_pose[3]
    ry = para_pose[4]
    rz = para_pose[5]
    rw = para_pose[6]
    quaternions = [rw, rx, ry, rz]

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=640, height=480)
    vis.add_geometry(pcd)

    ctr = vis.get_view_control()
    # get default camera parameters
    param = o3d.camera.PinholeCameraParameters()
    #param.intrinsic.intrinsic_matrix = intrinsic
    # get the pos x/y/z and quaternion frame by frame
   
    # create 4x4 transformation matrix
    extrinsic_matrix = np.eye(4)
    rotation_matrix = pcd.get_rotation_matrix_from_quaternion(quaternions)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = np.array(translation)
    param.extrinsic = extrinsic_matrix

        # set param to current camera control view
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.destroy_window()


def project2D(pcd, para_pose):
    translation = para_pose[:3]
    rx = para_pose[3]
    ry = para_pose[4]
    rz = para_pose[5]
    rw = para_pose[6]
    quaternions = [rw, rx, ry, rz] 


    #T = np.array([[-0.09047368913888931,-0.9589408040046692,-0.2687881588935852,0.09501823782920837],[0.9348847270011902, -0.17479585111141205,0.30892878770828247, -0.019299086183309555],[-0.34322744607925415, -0.22333601117134094, 0.9123135209083557, 1.0094271898269653],[0,0,0,1]])
    #pcd = pcd.transform(T)
    #rot2 = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    #pcd = pcd.rotate(rot2)

    #mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    pcd = pcd.translate(translation, relative=False)
    #rot = pcd.get_rotation_matrix_from_xyz((np.pi,0,0))
    #rot = R.from_quat(quaternions).as_matrix()
    rot = pcd.get_rotation_matrix_from_quaternion(quaternions)
    pcd = pcd.rotate(rot)
    
    

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    #Intrinsic camera parameter values:
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02 - 5
    cy_rgb = 2.5373616633400465e+02 - 12
    
    #Transform the colors from rgb to bgr convention:
    colors_rgb = colors
    colors_bgr = colors_rgb[..., ::-1].astype(np.float32)

    #Projection of the 3d-points onto the 2d-scene:
    eps=1e-6
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    #Lower threshhold of the Z-values (projection is undefined for Z=0)
    Z[Z < eps] = eps

    #Projection onto 2D-scene:
    u = (fx_rgb * X) / Z + cx_rgb
    v = (fy_rgb * Y) / Z + cy_rgb

    points_projected = np.zeros((u.shape[0], 2))
    points_projected[:,0] = u
    points_projected[:,1] = v

    return points_projected, colors_bgr

"""
Function, which creates an Image using the information from the projected 2D scene.


Input:
points: np.array where each row represents a point. Each row contains 3 values, corresponding to x, y, z coordinates.
colors: np.array where each row corresponds to the same row in points. Each row contains 3 values, which represent the color in BGR-convention.
height: int or None, specifies the height of the resulting picture
width: int or None, specifies the width of the resulting picture

Output:
image: np.array, size = (width, height, 3). Image representation of the 2D scene.
"""

def construct2DImage(points, colors, height = None, width = None, ):

 
    #Shifting the points to the positive axis
    points[:,0] = points[:,0]-np.min(points[:,0])
    points[:,1] = points[:,1]-np.min(points[:,1])

    points[:,0] = points[:,0]/np.max(points[:,0])*640
    points[:,1] = points[:,1]/np.max(points[:,1])*480

    #Scaling the distance between points
    points[:,0] = points[:,0]#/1000000
    points[:,1] = points[:,1]#/1000000

    #If height or width is None, the height and the width of the resulting picture will be calculated. 
    if (height == None) or (width == None):
        height = int(np.max(points[:,1]))#+10 
        width = int(np.max(points[:,0]))#+10 


    #The resulting image is being constructed:
    image = np.zeros((height, width, 3))
    for point, color in zip(points, colors):
        image_point = tuple(map(int, point))
        image[image_point[1]-1, image_point[0]-1,:] = color

   

    #Before returning, the color information is transformed to a format, which is compatible with cv2.
    return (image*255).astype(np.uint8)

def show_image(img: np.ndarray, title: str, save_image: bool = False, use_matplotlib: bool = False) -> None:
    """ Plot an image with either OpenCV or Matplotlib.

    :param img: :param img: Input image
    :type img: np.ndarray with shape (height, width) or (height, width, channels)

    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :type title: string

    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :type save_image: bool

    :param use_matplotlib: If this is set to True, Matplotlib will be used for plotting, OpenCV otherwise
    :type use_matplotlib: bool
    """

    # First check if img is color or grayscale. Raise an exception on a wrong type.
    if len(img.shape) == 3:
        is_color = True
    elif len(img.shape) == 2:
        is_color = False
    else:
        raise ValueError(
            'The image does not have a valid shape. Expected either (height, width) or (height, width, channels)')

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.

    elif img.dtype == np.float64:
        img = img.astype(np.float32)

    if use_matplotlib:
        plt.figure()
        plt.title(title)
        if is_color:
            # OpenCV uses BGR order while Matplotlib uses RGB. Reverse the the channels to plot the correct colors
            plt.imshow(img[..., ::-1])
        else:
            plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        cv2.imshow(title, img)
        cv2.waitKey(0)

    if save_image:
        if is_color:
            png_img = (cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) * 255.).astype(np.uint8)
        else:
            png_img = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA) * 255.).astype(np.uint8)
        cv2.imwrite(title.replace(" ", "_") + ".png", png_img)

def getGroundtruthPose(sceneID, pictureID):

    sceneID = "00"+str(sceneID)
    #pictureID = "0"*(5-len(str(pictureID)))+str(pictureID)

    current_path = Path(__file__).parent
    #Load Pointcloud
    df_poses = pd.read_csv(os.path.join(current_path, "scenes", "new", "scenes", sceneID, "groundtruth_handeye.txt"), delimiter=" ", header=None)
    return df_poses.iloc[pictureID-1,1:].to_numpy()
        
def get2DView(pcd, para_pose):
    # Load the point cloud
    #pcd = o3d.io.read_point_cloud("path_to_your_pointcloud.ply")

    fx = 5.1885790117450188e+02
    fy = 5.1946961112127485e+02
    cx = 3.2558244941119034e+02 - 5
    cy = 2.5373616633400465e+02 - 12


    # Define camera intrinsic parameters
    width, height = 640, 480
    #fx, fy = 525.0, 525.0  # Focal lengths
    #cx, cy = width / 2 -0.5, height / 2 -0.5 # Principal point

    intrinsic = o3d.camera.PinholeCameraIntrinsic(height, width, fx, fy, cx, cy)

    # Define camera extrinsic parameters (identity matrix for simplicity)

    translation = para_pose[:3]
    rx = para_pose[3]
    ry = para_pose[4]
    rz = para_pose[5]
    rw = para_pose[6]
    quaternions = [rx, ry, rz, rw] 


    T_object = np.array([[-0.09047368913888931,-0.9589408040046692,-0.2687881588935852,0.09501823782920837],[0.9348847270011902, -0.17479585111141205,0.30892878770828247, -0.019299086183309555],[-0.34322744607925415, -0.22333601117134094, 0.9123135209083557, 1.0094271898269653],[0,0,0,1]])
    translation_T = T_object[:3,3]
    rotation_T = T_object[:3,:3]
    
    #pcd = pcd.transform(T_object*T_scene)
    #pcd = pcd.translate(translation, relative=False)
    #rot = pcd.get_rotation_matrix_from_xyz((np.pi,0,0))
    #rot = R.from_quat(quaternions).as_matrix()
    #rot = pcd.get_rotation_matrix_from_quaternion(quaternions)
    #pcd = pcd.rotate(rot)
    rotation = R.from_quat(quaternions).as_matrix()

    T_scene = np.eye(4)
    T_scene[:3,:3] = rotation#np.matmul(rotation_T.T,rotation)
    T_scene[:3,3] = translation#+np.array([1.3,-0.8,0], dtype=np.float64)*0.1

    #pcd = pcd.transform(T_scene)
    #T_Camera = T_object#/T_scene

    T_Camera = np.dot(np.linalg.inv(T_scene),T_object)


    #rotation_C = rotation_T*rotation.T
    #translation_C = translation_T#+translation

# Get the rotation matrix
    #rotation_C = rotation_C.as_matrix()
    

    extrinsic = np.eye(4)

    #extrinsic[:3,:3] = rotation_C
    #extrinsic[:3,3] = translation_C

    extrinsic = T_Camera

    # Create a PinholeCameraParameters object
    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.intrinsic = intrinsic
    camera_parameters.extrinsic = extrinsic.astype(np.float64)

    # Create a renderer and render the point cloud to an image
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    
    ctr.convert_from_pinhole_camera_parameters(camera_parameters, allow_arbitrary=True)
    vis.poll_events()
    
    vis.update_renderer()

    # Capture the image
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    # Convert the image to a numpy array and display it
    image = np.asarray(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

para_pose = getGroundtruthPose(1,20)



pcd = loadObjectPointcloud("drill", use_voxel_downsampling=False)

get2DView(pcd, para_pose)


#points, colors = project2D(pcd, para_pose)
#img = construct2DImage(points, colors)
#show_image(img, "test", use_matplotlib=True)
