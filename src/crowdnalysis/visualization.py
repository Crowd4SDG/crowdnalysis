import numpy as np
from html_writer import Html

# TODO: Decide how to generate HTML

def html_description(consensus, data, picture_field):
    """
    returns an string with the HTML showing the pictures
    """
    best = np.argmax(consensus, axis=1)
    labels = np.unique(best)
    html = Html()
    for label in labels:
        # TODO: Get the name of the label by asking data
        html += "Images for label:"+str(label)
        task_indexes = np.where(best == label)
        task_picture_links = data.get_field(task_indexes, picture_field)
        for tpl in task_picture_links:
            html.print(tpl)

    return html