import pickle
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

plt.rcParams['font.size'] = 16

result_path='./exp1/all_results.pickle'

with open(result_path, 'rb') as file:
    all_results =pickle.load(file)

schemes=all_results.keys()
important_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
hook_points = ['attn.hook_q', 'attn.hook_k', 'attn.hook_v', 'attn.hook_z', 'mlp.hook_post', 'hook_resid_post']

layers_results= {}
all_acc={}
for c,scheme in enumerate(schemes):
    result=all_results[scheme]
    id_faith=result["ID Faithfulness"]
    ood_faith=result["OOD Faithfulness"]

    id_acc = result["ID Patched Accuracy"]
    ood_acc=result["OOD Patched Accuracy"]

    if c==0:
        if 'id' not in layers_results.keys():
            layers_results['id']= {}
            all_acc['id']={}
        for i,_ in enumerate(important_layers):
            for j,hp in enumerate(hook_points):
                if hp not in layers_results['id'].keys():
                    layers_results['id'][hp]=[]
                if hp not in all_acc['id'].keys():
                    all_acc['id'][hp]=[]
                layers_results['id'][hp].append(id_faith[i*6+j])
                all_acc['id'][hp].append(id_acc[i*6+j])
    if scheme not in layers_results.keys():
        layers_results[scheme]= {}
    if scheme not in all_acc.keys():
        all_acc[scheme]= {}
    for i, _ in enumerate(important_layers):
        for j, hp in enumerate(hook_points):
            if hp not in layers_results[scheme].keys():
                layers_results[scheme][hp] = []
            if hp not in all_acc[scheme].keys():
                all_acc[scheme][hp] = []
            layers_results[scheme][hp].append(ood_faith[i * 6 + j])
            all_acc[scheme][hp].append(ood_acc[i * 6 + j])


for hp in hook_points:
    plt.figure(figsize=(6,4))
    for k in layers_results.keys():
        plt.plot(important_layers,layers_results[k][hp],marker='.',label=k)
    plt.xlabel('Patching Layer')
    plt.ylabel('Faithfulness')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./%s/hp_%s.png'%('exp1',hp),dpi=200)
    # plt.show()

    plt.figure(figsize=(6, 4))
    for k in all_acc.keys():
        plt.plot(important_layers, all_acc[k][hp], marker='.', label=k)
    plt.xlabel('Patching Layer')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./%s/hpacc_%s.png' % ('exp1', hp), dpi=200)

