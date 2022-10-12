import os
import dominate
import numpy as np

def html_visualize(web_path, data, ids, cols, others=[], title='visualization', threading_num=10, version='predictive'):
    """Visualization in html.
    
    Args:
        web_path: string; directory to save webpage. It will clear the old data!
        data: dict; 
            key: {id}_{col}. 
            value: figure or text
                - figure: ndarray --> .png or [ndarrays] --> .gif
                - text: string or [string]
        ids: [string]; name of each row
        cols: [string]; name of each column
        others: (optional) [dict]; other figures
            - name: string; name of the data, visualize using h2()
            - data: string or ndarray(image)
            - height: (optional) int; height of the image (default 256)
        title: (optional) string; title of the webpage (default 'visualization')
        threading_num: (optional) int; number of threadings for imwrite (default 10)
    """
    with dominate.document(title=title) as web:
        dominate.tags.h1(title)
        with dominate.tags.table(border=1, style='table-layout: fixed;'):
            with dominate.tags.tr():
                with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', width='64px'):
                    dominate.tags.p('id')
                for col in cols:
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', ):
                        dominate.tags.p(col)
            for id in ids:
                with dominate.tags.tr():
                    bgcolor = 'F1C073' if id.startswith('train') else 'C5F173'
                    with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='center', bgcolor=bgcolor):
                        for part in id.split('_'):
                            dominate.tags.p(part)
                    for col in cols:
                        with dominate.tags.td(style='word-wrap: break-word;', halign='center', align='top'):
                            if f'{id}_{col}' in data:
                                value = data[f'{id}_{col}']
                                if col == 'relation-graph' or col == 'position-afford' or col == 'goal':
                                    dominate.tags.img(style='height:128px', src=value)
                                elif 'step' in col:
                                    dominate.tags.img(style='height:128px', src=value)
                                elif isinstance(value, str):
                                    dominate.tags.p(value)
                                elif isinstance(value, list) and isinstance(value[0], str):
                                    for v in value:
                                        dominate.tags.p(v)
                                elif isinstance(value, np.ndarray):
                                    for v in value:
                                        dominate.tags.p(np.array2string(v))
                                else:
                                    continue
    with open(os.path.join(web_path, 'index_{}.html'.format(version)), 'w') as fp:
        fp.write(web.render())