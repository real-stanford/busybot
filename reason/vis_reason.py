import numpy as np
from html_vis import html_visualize

def html(idx, edge_types, gt_edge_types, edge_type_after_gumbel_softmax_list, state_accurs,
               histories, future_actions, gt_predictions, predictions, vis_dir):
    ids = [str(i+1) for i in idx]
    cols = ['observation', 'action', 'prediction', 'gt-prediction', 'state-acc', 'raw-edge-type', 'other', 'gt-edge-type', 'scene-graph']
    data = dict()
    for index, id in enumerate(ids):
        data[f'{id}_observation'] = histories[index]
        data[f'{id}_action'] = future_actions[index]
        data[f'{id}_prediction'] = predictions[index]
        data[f'{id}_gt-prediction'] = gt_predictions[index]
        data[f'{id}_state-acc'] = state_accurs[index]

        data[f'{id}_raw-edge-type'] = edge_types[index]
        data[f'{id}_other'] = edge_type_after_gumbel_softmax_list[index]
        data[f'{id}_gt-edge-type'] = gt_edge_types[index]
        data[f'{id}_scene-graph'] = ''

    print(data.keys())
    # exit()

    # others = [
    #     {'name': 'other_figure_small', 'data': np.random.rand(32, 32, 3), 'height':128},
    #     {'name': 'other_figure_large', 'data': np.random.rand(32, 32, 3), 'height':256},
    #     {'name': 'other_string', 'data': 'text'},
    # ]

    html_visualize(
        web_path=vis_dir,
        data=data,
        ids=ids,
        cols=cols,
        title='Visualization',
        threading_num=4,
        image_folder=vis_dir
    )


# if __name__=='__main__':
#     main()
