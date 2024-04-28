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
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def extractPicture(image_path, load_size: int = 202, layer: int = 11, facet: str = 'key', remove_outliers: bool = False, model_type: str = 'dino_vits8', stride: int = 4, extractor: ViTExtractor = None):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #device = 'cpu'
        if extractor is None:
            extractor = ViTExtractor(model_type, stride, device=device)

        include_cls = remove_outliers

        #image_path = r'C:\Users\domin\Documents\RobotVisionProject\dino-vit-features\scenes\coffee.png'
        #Preprocess: Resizes Image, normalizes image, converts to rgb
        image_orig_name = Path(image_path).stem

        image_batch, image_pil = extractor.preprocess(image_path, load_size)
        descs = extractor.extract_descriptors(image_batch.to(device), layer, facet, bin, include_cls).cpu().detach().numpy()
        saliency_map = extractor.extract_saliency_maps(image_batch.to(device)).cpu().detach().numpy()
        curr_num_patches, curr_load_size = extractor.num_patches, extractor.load_size
        if remove_outliers:
            cls_descriptor, descs = torch.from_numpy(descs[:, :, 0, :]), descs[:, :, 1:, :]
        else:
            cls_descriptor = None

        return descs, saliency_map, curr_num_patches, curr_load_size, image_pil, cls_descriptor, image_orig_name


def extractPicturesFromDirectory(directory_path, num_pictures: int = -1, load_size: int = 202, layer: int = 11, facet: str = 'key', remove_outliers: bool = False, model_type: str = 'dino_vits8', stride: int = 4):

    descriptors_list = []
    saliency_maps_list = []
    image_pil_list = []
    num_patches_list = []
    load_size_list = []
    image_orig_name_list= []

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
        
        descs, saliency_map, curr_num_patches, curr_load_size, image_pil, cls_descriptor, image_orig_name = extractPicture(image_path, load_size, layer, facet, remove_outliers, model_type, stride, extractor)

        num_patches_list.append(curr_num_patches)
        load_size_list.append(curr_load_size)
        image_pil_list.append(image_pil)
        image_orig_name_list.append(image_orig_name)

        if remove_outliers:
            cls_descriptors.append(cls_descriptor)
        else:
            cls_descriptors = None

        descriptors_list.append(descs)
        saliency_maps_list.append(saliency_map)

    return descriptors_list, saliency_maps_list, num_patches_list, load_size_list, image_pil_list, cls_descriptors, image_orig_name_list


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


    labels_list = []
    i = 0
    
    for image, num_patches in zip(pca_per_image, num_patches_list):
        i = i+1
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
    

    #Create a Region Adjacency Graph from the patch ID array. (Each unique number creates a node).
    G = ski.graph.RAG(descr_label_array, connectivity=2)

    for n in G.nodes():
        G.nodes[n].update({"labels" :[n]})

    maxweight = 0
    for u,v in G.edges():
        source_descr_idx = np.where(descr_label_array==u)
        source_descriptor = torch.from_numpy(descriptors[source_descr_idx[0], source_descr_idx[1]])
        target_descr_idx = np.where(descr_label_array==v)
        target_descriptor = torch.from_numpy(descriptors[target_descr_idx[0], target_descr_idx[1]])
        cos_sim = torch.nn.CosineSimilarity(dim = 1)
        weight = cos_sim(source_descriptor, target_descriptor)
        G.edges[u,v]["weight"] = 1-weight.numpy()[0]
        G.edges[u,v]["count"] = 1
        if (1-weight.numpy()[0]) > maxweight:
            maxweight = 1-weight.numpy()[0]
        
    return G, descr_label_array, maxweight

def applyNormalizedCut(G, labels, image_pil, thresh, maxweight):
    new_labels = ski.graph.cut_normalized(labels, G, thresh=thresh, max_edge = maxweight)




    out = ski.color.label2rgb(new_labels, np.asarray(image_pil), kind='avg', bg_label=0)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
    ax[0].imshow(image_pil)
    ax[1].imshow(out)
    return new_labels


def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']

    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst) / count,
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


def plotLabelsImage(labels, image_pil, kind='avg', save_image = False, save_path=None, title=""):
    labels_pil = Image.fromarray(labels.astype(dtype=np.uint32))
    labels_resized = np.array(labels_pil.resize(image_pil.size, resample=Image.NEAREST), dtype=np.uint32)

    out = ski.color.label2rgb(labels_resized, np.asarray(image_pil), kind=kind, bg_label=0)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
    ax[0].axis('off')
    ax[0].imshow(image_pil)
    ax[1].axis('off')
    ax[1].imshow(out)

    ax[0].set_title("Original")
    ax[1].set_title(title)

    if save_image:
        if save_path != None:
            plt.savefig(fname=save_path)
        else:
            print("Specify a savepath!")
    plt.close()
    

def plotGraph(G, labels, image_pil, save_image = False, save_path=None):
    labels_pil = Image.fromarray(labels.astype(dtype=np.uint32))
    labels_resized = np.array(labels_pil.resize(image_pil.size, resample=Image.NEAREST), dtype=np.uint32)

    fig, ax = plt.subplots(nrows = 1, sharex=True, sharey=True, figsize=(6,8))
    ax.set_title("Region Adjacency Graph")
    lc = ski.graph.show_rag(labels_resized, G, np.asarray(image_pil), ax=ax)
    fig.colorbar(lc, fraction = 0.03, ax=ax)
    ax.axis('off')
    #plt.show()

    if save_image:
        if save_path != None:
            plt.savefig(fname=save_path)
        else:
            print("Specify a savepath!")
    plt.close()


def plotSaliencyMap(saliency_map, image_pil, num_patches, save_image=False, save_path = None):
    saliency_map = np.squeeze(saliency_map.reshape((num_patches[0], num_patches[1], -1)))
    cmap = plt.get_cmap("jet")
    image = (cmap((saliency_map*255).astype(np.uint8))[:,:,:-1]*255).astype(np.uint8)
    saliency_pil = Image.fromarray(image)
    saliency_resized = np.array(saliency_pil.resize(image_pil.size, resample=Image.NEAREST), dtype=np.uint8)


    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))
    ax[0].axis('off')
    ax[0].imshow(image_pil)
    ax[1].axis('off')
    ax[1].imshow(saliency_resized)

    ax[0].set_title("Original")
    ax[1].set_title("Saliency Map")

    if save_image:
        if save_path != None:
            plt.savefig(fname=save_path)
        else:
            print("Specify a savepath!")
    plt.close()


def applyMergeHierarchical(G, labels, image_pil, thresh):
 

    labels_new = ski.graph.merge_hierarchical(
    labels,
    G,
    thresh=thresh,
    rag_copy=True,
    in_place_merge=True,
    merge_func=merge_boundary,
    weight_func=weight_boundary,
    )

    return labels_new
    
def filterSalientLabels(labels_new, saliency_map, num_patches, image_pil, thresh = 0.5):
    

    saliency_map = np.squeeze(saliency_map.reshape((num_patches[0], num_patches[1], -1)))    
    unique_labels = list(np.unique(labels_new))
    labels_salient = np.zeros(labels_new.shape, dtype=labels_new.dtype)

    label_saliency_list = []
    
    for label in unique_labels:
        label_saliency_list.append(saliency_map[labels_new == label].mean())
    
    mean = np.mean(label_saliency_list)
    std_dev = np.std(label_saliency_list)

    
    for label, label_saliency in zip(unique_labels, label_saliency_list):
        if label_saliency > (mean+std_dev):
            labels_salient[np.where(labels_new==label)] = label
    

    return labels_salient

    
def applyGrabCut(labels_salient, image_pil, num_patches, load_size):
    #mask = np.isin(labels, labels_salient).reshape(num_patches)
    resized_mask = np.array(Image.fromarray(labels_salient.astype(dtype=np.uint32)).resize((load_size[1], load_size[0]), resample=Image.LANCZOS))
    try:
        # apply grabcut on mask
        grabcut_kernel_size = (7, 7)
        kernel = np.ones(grabcut_kernel_size, np.uint8)
        forground_mask = cv2.erode(np.uint8(resized_mask), kernel)
        forground_mask = np.array(Image.fromarray(forground_mask).resize(image_pil.size, Image.NEAREST))
        background_mask = cv2.erode(np.uint8(1 - resized_mask), kernel)
        background_mask = np.array(Image.fromarray(background_mask).resize(image_pil.size, Image.NEAREST))
        full_mask = np.ones((load_size[0], load_size[1]), np.uint8) * cv2.GC_PR_FGD
        full_mask[background_mask == 1] = cv2.GC_BGD
        full_mask[forground_mask == 1] = cv2.GC_FGD
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(np.array(image_pil), full_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        grabcut_mask = np.where((full_mask == 2) | (full_mask == 0), 0, 1).astype('uint8')
    except Exception:
        # if mask is unfitted from gb (e.g. all zeros) -- don't apply it
        grabcut_mask = resized_mask.astype('uint8')

    #grabcut_mask = Image.fromarray(np.array(grabcut_mask, dtype=bool))
    return grabcut_mask

scenes = 6
num_pictures = 20
scenes_directory = r'C:\Users\domin\Documents\RobotVisionProject\dino-vit-features\scenes\new\scenes'
for scene in range(scenes):
    print(str(scene+1)+"/"+str(scenes))
    scene_directory = scenes_directory+"\\00"+str(scene+1)+"\\rgb"
    save_dir = scene_directory+"\\results\\"

    if Path.exists(Path(save_dir)) == False:
        Path.mkdir(save_dir)

    image_directory_path = Path(scene_directory)
    #extractPicture(image_directory+"\\"+"000001.png")

    

    descriptors_list, saliency_maps_list, num_patches_list, load_size_list, image_pil_list, cls_descriptors, image_orig_name_list = extractPicturesFromDirectory(scene_directory, num_pictures, 202)#202)
    #plt.ion()
    i = 0
    for descriptor, saliency_map, num_patches, load_size, image_pil, image_orig_name in tqdm(zip(descriptors_list, saliency_maps_list, num_patches_list, load_size_list, image_pil_list, image_orig_name_list)):
        i+=1
        print(f"working on {i}/{num_pictures}")
        descriptor = np.squeeze(descriptor)

        save_path = save_dir+image_orig_name+"_saliencyMap.png"
        plotSaliencyMap(saliency_map, image_pil, num_patches, save_image=True, save_path = save_path)
        
        

        graph, labels, maxweight = constructGraph(descriptor, num_patches, load_size, image_pil)
        save_path = save_dir+image_orig_name+"_graph.png"
        plotGraph(graph, labels, image_pil, save_image = True, save_path=save_path,)
    
        labels_new = applyMergeHierarchical(graph, labels, image_pil, thresh=0.28)
        save_path = save_dir+image_orig_name+"_mergeHierarchical.png"
        plotLabelsImage(labels_new, image_pil, save_image = True, save_path=save_path, title="After hierarchical merging")

        labels_salient = filterSalientLabels(labels_new, saliency_map, num_patches, image_pil)
        save_path = save_dir+image_orig_name+"_salientLabels.png"
        plotLabelsImage(labels_salient, image_pil, save_image = True, save_path=save_path, title="After filtering out unsalient labels")
        

        grabcut_mask = applyGrabCut(labels_salient, image_pil, num_patches, load_size)
        save_path = save_dir+image_orig_name+"_grabcut.png"
        plotLabelsImage(grabcut_mask, image_pil, kind="overlay", save_image = True, save_path=save_path, title="After applying grabcut")
    
    del descriptors_list
    del saliency_maps_list
    del num_patches_list
    del load_size_list
    del image_pil_list
    del cls_descriptors
    del image_orig_name_list