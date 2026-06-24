import torch
import shutil
import os
from PIL import Image
import json # Added for JSON output

import torch.nn as nn
import torchvision.transforms.functional as tf
import argparse
import importlib.util
import os


def load_model_from_file(model_path, class_name):
    spec = importlib.util.spec_from_file_location(class_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class = getattr(module, class_name)
    return model_class()


def readImg(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = tf.to_tensor(img)
    img = tf.rgb_to_grayscale(img, num_output_channels=3) # This might be unusual for ImageNet pretrained models, usually RGB
    img = tf.normalize(img, (0.485, 0.456, 0.406),
                       (0.229, 0.224, 0.225)).unsqueeze(0)
    return img

def featurize_images(folder, model):
    fImg_name = []
    img_f = []

    model.eval()
    for fRoot, fDirs, fFiles in os.walk(folder):
        for ffile in fFiles:
            fImg = readImg(os.path.join(fRoot, ffile))
            if fImg is None:
                continue
            fImg_name.append(os.path.join(fRoot, ffile).replace("\\", "/"))
            with torch.no_grad():
                fImg = fImg.to(device="cuda")
                f_vec = nn.functional.normalize(model(fImg))

                img_f.append(f_vec)

    img_f_mat = torch.cat(img_f, dim=0)
    return img_f_mat, fImg_name

def cos_similarity(f_mat_src, f_mat_target):
    # cosine similarity of feature vectors
    score = torch.matmul(f_mat_src, f_mat_target.T)
    return score

def match(model, source, target, topk, output_folder_path):
    model.eval()

    source_name = []
    source_f = []

    # output_folder = output_folder_path != '' # Original line, now output_folder_path is always used

    for sRoot, sDirs, sFiles in os.walk(source):
        for sfile in sFiles:
            sImg = readImg(os.path.join(sRoot, sfile))
            if sImg is None:
                continue
            source_name.append(os.path.join(sRoot, sfile).replace("\\", "/"))
            with torch.no_grad():
                sImg = sImg.to(device="cuda")
                f_vec3 = nn.functional.normalize(model(sImg))

                source_f.append(f_vec3)

    source_f_mat = torch.cat(source_f, dim=0)

    target_name = []
    target_f = []

    for tRoot, tDirs, tFiles in os.walk(target):
        for tfile in tFiles:
            tImg = readImg(os.path.join(tRoot, tfile))
            if tImg is None:
                continue
            target_name.append(os.path.join(tRoot, tfile).replace("\\", "/"))
            with torch.no_grad():
                tImg = tImg.to(device="cuda")
                f_vec3 = nn.functional.normalize(model(tImg))
                target_f.append(f_vec3)

    target_f_mat = torch.cat(target_f, dim=0)
    score_mat = cos_similarity(source_f_mat, target_f_mat)

    # Create lists of base filenames without extensions
    source_basenames = [os.path.splitext(os.path.basename(s_path))[0] for s_path in source_name]
    target_basenames = [os.path.splitext(os.path.basename(t_path))[0] for t_path in target_name]

    # Set similarity score to a very low value for self-matches
    for i, s_basename in enumerate(source_basenames):
        for j, t_basename in enumerate(target_basenames):
            if s_basename == t_basename:
                score_mat[i, j] = float("-inf")  # Set to negative infinity to ensure it's never a topk match

    max_score, max_arg = torch.topk(score_mat, k=topk, dim=1)

    # acc_k = [] # Commented out: No longer needed for evaluation metrics

    for i, ip in enumerate(max_arg):
        # passed = False # Commented out: No longer needed for evaluation metrics
        # print('For Image {}:'.format(source_name[i])) # Commented out: Original debug print
        si_full_path = source_name[i]
        si_name_without_ext = os.path.splitext(os.path.basename(si_full_path))[0]

        # Commented out original image copying logic
        # if output_folder:
        #     os.makedirs('{}/{}'.format(match_name, i))
        #     shutil.copyfile(
        #         source_name[i], '{}/{}/source_{}'.format(match_name, i, si_name))

        # t_names = [] # Commented out: No longer needed for evaluation metrics

        source_match_data = {
            "source": si_full_path # Added source image path
        }
        for k in range(topk):
            # ti_name = target_name[ip[k]].split('/')[-1] # Commented out: Original target name extraction
            # t_names.append(ti_name) # Commented out: No longer needed for evaluation metrics

            # Commented out original image copying logic
            # if output_folder:
            #     shutil.copyfile(
            #         target_name[ip[k]], '{}/{}/{}'.format(match_name, i, ti_name))
            # # else:
            # #    print(ti_name)

            # if ti_name == si_name:
            #     passed = True
            #     acc_k.append(k+1)
            #     break
            
            target_filepath = target_name[ip[k]]
            similarity_score = max_score[i][k].item()

            source_match_data[str(k+1)] = {
                "filepath": target_filepath,
                "similarity_score": similarity_score
            }
        
        # Save the JSON file for the current source image
        if output_folder_path:
            os.makedirs(output_folder_path, exist_ok=True) # Ensure the output folder exists
            json_filename = f"{si_name_without_ext}.json"
            json_filepath = os.path.join(output_folder_path, json_filename)
            with open(json_filepath, 'w') as f:
                json.dump(source_match_data, f, indent=4)
            print(f"Saved matches for {os.path.basename(si_full_path)} to {json_filepath}")

        # Commented out evaluation metrics display
        # if passed:
        #    print('Pass')
        # else:
        #  for path in t_names:
        #    print(path)

    # Commented out evaluation metrics calculation and print
    # for k in range(topk):
    #     count = len([i for i in acc_k if i <= k+1])
    #     print('Top {} Eval Acc: {}/{}'.format(k+1, count, max_arg.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights', type=str,
        default='weights_2-11_199.pt', help='initial weights path')
    parser.add_argument(
        '--model-definition', type=str,
        default='ModelCombo.py', help='path to the model definition file')
    parser.add_argument('--topk', type=int, default=15)
    parser.add_argument('--source', type=str, default='./eval')
    parser.add_argument('--target', type=str, default='./target')
    parser.add_argument('--output-folder', type=str, default='')
    opt = parser.parse_args()

    model_filename = os.path.basename(opt.model_definition)
    model_class_name = os.path.splitext(model_filename)[0]
    model = load_model_from_file(opt.model_definition, model_class_name)

    model.load_state_dict(torch.load(opt.weights), strict=False)
    gpu = True
    if gpu:
        model = model.to(device="cuda")

    match(model, opt.source, opt.target, opt.topk, opt.output_folder)
