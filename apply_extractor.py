from extractor import ViTExtractor
import torch
import time
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from typing import List, Tuple
from pca import plot_pca
from tqdm import tqdm
from pathlib import Path
import networkx as nwx
import skimage as ski


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def extractPicture(image_path, load_size: int = 202, layer: int = 11, facet: str = 'key', remove_outliers: bool = False, model_type: str = 'dino_vits8', stride: int = 4, extractor: ViTExtractor = None):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if extractor is None:
            extractor = ViTExtractor(model_type, stride, device=device)

        include_cls = remove_outliers

        #Preprocess: Resizes Image, normalizes image, converts to rgb
        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls).cpu().detach().numpy()
        saliency_map = extractor.extract_saliency_maps(image_batch.to(device)).cpu().detach().numpy()
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
        if remove_outliers:
            cls_descriptor, descs = torch.from_numpy(descs[:, :, 0, :]), descs[:, :, 1:, :]
        else:
            cls_descriptor = None

        return descs, saliency_map, curr_num_patches, curr_load_size, image_pil, cls_descriptor


def extractPicturesFromDirectory(directory_path, num_pictures: int = -1, load_size: int = 202, layer: int = 11, facet: str = 'key', remove_outliers: bool = False, model_type: str = 'dino_vits8', stride: int = 4):

    descriptors_list = []
    saliency_maps_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []

    if remove_outliers:
        cls_descriptors = []

    include_cls = remove_outliers

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)

    start_time = time.time()
    #Preprocess: Resizes Image, normalizes image, converts to rgb
    i = 0
    for image_name in os.listdir(directory_path):
        
        
        if image_name.endswith(".png") is False:
            continue
        i+=1
        
        if (num_pictures != -1) and (i == (num_pictures+1)):
            break
        print(i)
        image_path = directory_path+"\\"+image_name
        
        descs, saliency_map, curr_num_patches, curr_load_size, image_pil, cls_descriptor = extractPicture(image_path, load_size, layer, facet, remove_outliers, model_type, stride, extractor)

        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        image_pil_list.append(image_pil)

        if remove_outliers:
            cls_descriptors.append(cls_descriptor)
        else:
            cls_descriptors = None

        descriptors_list.append(descs)
        saliency_maps_list.append(saliency_map)

    return descriptors_list, saliency_maps_list, num_patches_list, load_size_list, image_pil_list, cls_descriptors


def applyPCA(descriptors_list, n_components: int = 4, all_together: bool = True) -> List[Tuple[Image.Image, np.ndarray]]:

    if all_together:
        descriptors = np.concatenate(descriptors_list, axis=2)[0, 0]
        pca = PCA(n_components=n_components).fit(descriptors)
        pca_descriptors = pca.transform(descriptors)
        split_idxs = np.array([num_patches[0] * num_patches[1] for num_patches in num_patches_list])
        split_idxs = np.cumsum(split_idxs)
        pca_per_image = np.split(pca_descriptors, split_idxs[:-1], axis=0)
    else:
        pca_per_image = []
        for descriptors in descriptors_list:
            pca = PCA(n_components=n_components).fit(descriptors[0, 0])
            pca_descriptors = pca.transform(descriptors[0, 0])
            pca_per_image.append(pca_descriptors)
    results = [(pil_image, img_pca.reshape((num_patches[0], num_patches[1], n_components))) for
               (pil_image, img_pca, num_patches) in zip(image_pil_list, pca_per_image, num_patches_list)]
    return results


def applyDBScan(pca_per_image, num_patches_list):

    """
    labels_list = []

    for image, num_patches in zip(pca_per_image, num_patches_list):
        pca_image = image[1]
        print(pca_image)
        print(pca_image[0])
        X = pca_image.reshape(-1, pca_image.shape[2])
        clustering = DBSCAN(eps=0.8, min_samples=4).fit(X)
        labels_list.append(clustering.labels_.reshape(num_patches[0], num_patches[1]))
    """

    labels_list = []
    i = 0
    
    for image, num_patches in zip(pca_per_image, num_patches_list):
        i = i+1
        #X = pca_image.reshape(-1, pca_image.shape[2])
        clustering = DBSCAN(eps=15, min_samples=5).fit(image[0,0])
        labels_list.append(clustering.labels_.reshape(num_patches[0], num_patches[1]))

    return labels_list

def applyDBScanPCA(pca_per_image, num_patches_list):

   
    labels_list = []

    for image, num_patches in zip(pca_per_image, num_patches_list):
        pca_image = image[1]
        X = pca_image.reshape(-1, pca_image.shape[2])
        clustering = DBSCAN(eps=0.8, min_samples=4).fit(X)
        labels_list.append(clustering.labels_.reshape(num_patches[0], num_patches[1]))
    return labels_list
    


def plotClusters(pil_image, labels_image, save_dir, save_prefix, clustering_name):
    cmap = plt.get_cmap("jet")

    save_dir = Path(save_dir)
    comp = labels_image[:, :]
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)
    comp_file_path = save_dir / f'{save_prefix}_{clustering_name}.png'
    #clustering_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
    image = (cmap((comp_img*255).astype(np.uint8))[:,:,:-1]*255).astype(np.uint8)
    clustering_pil = Image.fromarray(image)
    clustering_pil = clustering_pil.resize(pil_image.size, resample=Image.NEAREST)
    
    clustering_pil.save(comp_file_path)


def constructGraph(descriptors, num_patches, load_size, pil_image):
    #Reshape descriptor array, such that the dimensions agree with the size of the sampled image (from the ViT)
    descriptors = descriptors.reshape((num_patches[0], num_patches[1], -1))

    #Create an array in which the patches got a number assigned ("patch ID"). 
    descr_label_array = np.zeros((num_patches[0], num_patches[1]), dtype=np.uint32)
    i = 0
    for y in range(num_patches[0]):
        for x in range(num_patches[1]):
            i+=1
            descr_label_array[y,x] = i
    
    #Resize the array with the patch ID such that is the same size as the input picture of the ViT.
    #This new array will later be used as a lookup table.
    descr_label_pil = Image.fromarray(descr_label_array)
    descr_label_array_resized = np.array(descr_label_pil.resize(pil_image.size, resample=Image.NEAREST), dtype=np.uint32)

    #Create a Region Adjacency Graph from the patch ID array. (Each unique number creates a node).
    G = ski.graph.RAG(descr_label_array_resized, connectivity=2)

    for n in G.nodes():
        G.nodes[n].update({"labels" :[n]})

    for u,v in G.edges():
        source_descr_idx = np.where(descr_label_array==u)
        source_descriptor = torch.from_numpy(descriptors[source_descr_idx[0], source_descr_idx[1]])
        target_descr_idx = np.where(descr_label_array==v)
        target_descriptor = torch.from_numpy(descriptors[target_descr_idx[0], target_descr_idx[1]])
        cos_sim = torch.nn.CosineSimilarity(dim = 1)
        weight = cos_sim(source_descriptor, target_descriptor)
        G.edges[u,v]["weight"] = 1-weight.numpy()[0]


    ski.graph.show_rag(descr_label_array_resized, G, np.asarray(pil_image))
    return G, descr_label_array_resized

def applyNormalizedCut(G, labels, thresh):
    new_labels = ski.graph.cut_normalized(labels, G, thresh=thresh)


    #color.label2rgb(labels2, img, kind='avg', bg_label=0

    cmap = plt.get_cmap("viridis")
    comp = new_labels[:, :]
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)

    image = (cmap((comp_img*255).astype(np.uint8))[:,:,:-1]*255).astype(np.uint8)
    normalized_cut_pil = Image.fromarray(image)
    normalized_cut_pil.show()
    
    

image_directory = r'C:\Users\domin\Documents\RobotVisionProject\dino-vit-features\scenes\new\scenes\001\rgb'
save_dir = r'C:\Users\domin\Documents\RobotVisionProject\dino-vit-features\scenes\j_002\pca'

image_directory_path = Path(image_directory)
#extractPicture(image_directory+"\\"+"000001.png")
descriptors_list, saliency_maps_list, num_patches_list, load_size_list, image_pil_list, cls_descriptors = extractPicturesFromDirectory(image_directory, 1,202)
plt.ion()
for descriptor, num_patches, load_size, image_pil in zip(descriptors_list, num_patches_list, load_size_list, image_pil_list):
    descriptor = np.squeeze(descriptor)


    graph, labels = constructGraph(descriptor, num_patches, load_size, image_pil)
    applyNormalizedCut(graph, labels, thresh=0.06)
    applyNormalizedCut(graph, labels, thresh=0.05)
    applyNormalizedCut(graph, labels, thresh=0.04)
    applyNormalizedCut(graph, labels, thresh=0.03)
    applyNormalizedCut(graph, labels, thresh=0.02)





print(descriptors_list)
"""
pca_per_image = applyPCA(descriptors_list)
images_paths = [x for x in image_directory_path.iterdir() if x.suffix.lower() in ['.jpg', '.png', '.jpeg']]

for image_path, (pil_image, pca_image) in tqdm(zip(images_paths, pca_per_image)):
    save_prefix = image_path.stem#[:image_path.find(".")]
    plot_pca(pil_image, pca_image, str(save_dir), save_prefix=save_prefix)

labels_list = applyDBScan(descriptors_list, num_patches_list)
labels_PCA_list = applyDBScanPCA(pca_per_image, num_patches_list)

for image_path, cluster_image, pil_image in tqdm(zip(images_paths, labels_list, image_pil_list)):
    save_prefix = image_path.stem
    plotClusters(pil_image, cluster_image, str(save_dir), save_prefix, "DBSCAN")

for image_path, cluster_image, pil_image in tqdm(zip(images_paths, labels_PCA_list, image_pil_list)):
    save_prefix = image_path.stem
    plotClusters(pil_image, cluster_image, str(save_dir), save_prefix, "PCA_DBSCAN")

print("Done"+str(time.time()))
"""