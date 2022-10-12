import numpy as np
import os

from html_vis import html_visualize

def html(num_evaluation, phase, version, data_folder):
    ids = [str(id+1) for id in range(num_evaluation)]
    data = dict()
    cols = ['relation-graph', 'position-afford', 'init', 'goal']
    cols += [f'step-{i+1}' for i in range(8)]
    for task_id in ids:
        data[f'{task_id}_relation-graph'] = os.path.join('../', str(task_id), 'scene_graph.png')
        data[f'{task_id}_position-afford'] = os.path.join('../', str(task_id), 'affordance_map.png')
        data[f'{task_id}_init'] = os.path.join(str(task_id), 'init.png')
        data[f'{task_id}_goal'] = os.path.join(str(task_id), 'goal.png')
        num_steps = 0
        for file in os.listdir(os.path.join('/proj/crv/zeyi/busybot/plan/vis/{}/{}/planning-{}'.format(phase, data_folder, version), str(task_id))):
            if(file.startswith('step')):
                num_steps += 1
        for i in range(num_steps):
            data[f'{task_id}_step-{i+1}'] = os.path.join(str(task_id), f'step_{i+1}.png')

    # print(data.keys())

    html_visualize(
        web_path=f'/proj/crv/zeyi/busybot/plan/vis/{phase}/{data_folder}/planning-{version}',
        data=data,
        ids=ids,
        cols=cols,
        title='Visualization',
        threading_num=4,
        version=version
    )
