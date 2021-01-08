import numpy as np
from dominate import document, tags
from dominate.util import raw

from .data import Data

def html_description(consensus, data: Data, question, picture_field, width="120", height="90",
                     dec="3", warn_threshold=.1, output_file=None):
    """
    returns a string with the HTML showing the pictures
    """
    best = np.argmax(consensus, axis=1)
    labels = np.unique(best)
    label_names = list(data.df[question].cat.categories)

    FRAME_COLOR = "#ef0707"  # "#e30c0c"

    style_ = """
        .labels {{
          overflow-x: scroll;
          overflow-y: hidden;
          white-space: nowrap;
          -webkit-overflow-scrolling: touch;
        }}
        .task-image {{
          display: inline-block;
        }}
        .task-image img {{
          max-width: {w}px;
          max-height: {h}px;
        }}
        .warn{{
          padding:2px;
          border:6px solid {c};
         }}
    """
    style_ = style_.format(w=width, h=height, c=FRAME_COLOR)

    title_ = "{} Consensus :: {}".format(data.data_src, question.title())
    with document(title=title_) as doc:
        with doc.head:
            tags.style(style_)
            tags.base(target="_blank")
        with doc.body as body:
            body.add(tags.h1(title_, style="color:Tomato; text-align:center"))
            body.add(tags.p(raw("- Hover on any image to read its best and second best consensus probabilities.</br>"
                                "- When these two probabilities have a difference &le; {t}, the image is framed with "
                                "<span style='color:{c}; font-weight:bold'>borders</span> and marked as warning.</br>"
                                "- Scroll horizontally to view all images.".format(t=warn_threshold, c=FRAME_COLOR))))
            for label in labels:
                task_indices = np.where(best == label)[0]
                task_picture_links = data.get_field(task_indices, picture_field, unique=True)
                label_id = "label" + str(label)
                body.add(tags.h2("{} ({}):".format(str(label_names[label]).title(), len(task_indices)), id=label_id))
                n_warn = 0
                with body.add(tags.div(cls='labels')):
                    for ix, tpl in enumerate(task_picture_links):
                        l = tags.div(cls="task-image")
                        task_index = task_indices[ix]
                        probabilities = [(p, li) for li, p in enumerate(consensus[task_index])]
                        probabilities.sort(key=lambda x: x[0], reverse=True)
                        probs_str = "\n".join("- {0}: {1:.{2}f}".format(
                            label_names[li], p, dec) for p, li in probabilities[:2])
                        img_title = "Task Index: {}\n{}".format(str(task_index), probs_str)
                        if probabilities[0][0] - probabilities[1][0] <= warn_threshold:
                            n_warn += 1
                            kwargs = {"cls": "warn"}
                        else:
                            kwargs = {}
                        l += tags.a(tags.img(src=tpl, title=img_title, **kwargs), href=tpl)
                    if n_warn > 0:
                        label_header = doc.body.getElementById(label_id)
                        label_header.add_raw_string(" (<span style='color:{c}'>{n}</span> warning{s})".format(
                            c=FRAME_COLOR, n=str(n_warn), s="s" if n_warn > 1 else ""))
    if output_file:
        with open(output_file, 'w') as f:
            f.write(doc.render())
        print("Rendered HTML for the consensus of question '{}' is saved into file:\n '{}'".format(question, output_file))
    return str(doc)
