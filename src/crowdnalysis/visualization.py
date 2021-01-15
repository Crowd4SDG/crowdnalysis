import os
from typing import List, Union

import numpy as np
import pandas as pd
from dominate import document, tags, util

from .data import Data


def html_description(consensus: np.ndarray, data: Data, question: str, picture_field: str, width=120, height=90,
                     dec=3, warn_threshold=.1, output_file: str = None, pretty=True) -> str:
    """Returns an HTML string that displays the images of tasks.

    Optionally, saves the string to an HTML file.

    Args:
        consensus: Consensus probabilities of tasks
        data : Annotation data
        question: Question for annotation
        picture_field: column name in the `Data.df` dataframe
        width: max image width
        height: max image height
        dec: number of decimal digits of probabilities to display
        warn_threshold: threshold difference between best and second best consensus probabilities to be marked
            as warning
        output_file: (optional) full path to the output HTML file
        pretty: True, for a more human readable HTML string; False, for a smaller sized file.

    Returns:
        HTML string

    """

    FRAME_COLOR = "#ef0707"
    STYLE = """
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
    """.format(w=width, h=height, c=FRAME_COLOR)
    NOTES = ("- Hover on any image to read its best and second best consensus probabilities.<br/>"
             "- When these two probabilities have a difference &le; {t}, the image is framed with "
             "<span style='color:{c}; font-weight:bold'>borders</span> and marked as warning.<br/>"
             "- Scroll horizontally to view all images.").format(t=warn_threshold, c=FRAME_COLOR)
    TITLE = "{} Consensus :: {}".format(data.data_src, question.title())

    best = np.argmax(consensus, axis=1)
    labels = np.unique(best)
    label_names = list(data.df[question].cat.categories)

    with document(title=TITLE) as doc:
        with doc.head:
            tags.style(STYLE)
            tags.base(target="_blank")
        with doc.body as body:
            body.add(tags.h1(TITLE, style="color:Tomato; text-align:center"))
            body.add(tags.p(util.raw(NOTES)))
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
                        probabilities = [(p, li) for li, p in enumerate(consensus[task_index, :])]
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
    html_str = doc.render(pretty=pretty, xhtml=True)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(html_str)
        print("HTML for the {} consensus for the question '{}' is saved into file:\n '{}'".format(
            data.data_src, question, os.path.relpath(output_file)))
    return html_str


def csv_description(consensus: np.ndarray, data: Data, question: str, picture_field: str, dec=3,
                    output_file: str = None) -> Union[None, str]:
    """Saves/returns the CSV-format representation of the consensus on tasks.

    `diff_best_two` column is the difference between the best and second best consensus probabilities.

    Args:
        consensus: Consensus probabilities of tasks
        data : Annotation data
        question: Question for annotation
        picture_field: column name in the `Data.df` dataframe
        dec: number of decimal digits of probabilities to display
        output_file: (optional) full path to the output HTML file

    Returns:
        If output_file is None, returns the resulting csv format as a string. Otherwise, returns None.
    """
    def diff_best_two(x):
        probabilities = list(x)
        probabilities.sort(reverse=True)
        return probabilities[0] - probabilities[1]

    label_names = list(data.df[question].cat.categories)
    df = pd.DataFrame(consensus, columns=label_names)
    df["diff_best_two"] = df.apply(diff_best_two, axis=1)
    df.index.name = "task_index"
    df[picture_field] = data.get_field(list(df.index), picture_field, unique=True)
    df = round(df, dec)
    val = df.to_csv(output_file)
    if output_file:
        print("CSV-format output for the {} consensus for the question '{}' is saved into file:\n '{}'".format(
            data.data_src, question, os.path.relpath(output_file)))
    return val


def html_description_crossed(consensus: np.ndarray, data: Data, question: str, pipeline_task_indices: List[int],
                             picture_field: str, width=120, height=90,
                             dec=3, warn_threshold=.1, output_file: str = None, pretty=True) -> str:
    """Returns an HTML string that displays the images of tasks.

    Optionally, saves the string to an HTML file.

    Args:
        consensus: Consensus probabilities of tasks
        data : Annotation data
        question: Question for annotation
        pipeline_task_indices: List of task indices of that were filtered by POLIMI Pipeline
        picture_field: column name in the `Data.df` dataframe
        width: max image width
        height: max image height
        dec: number of decimal digits of probabilities to display
        warn_threshold: threshold difference between best and second best consensus probabilities to be marked
            as warning
        output_file: (optional) full path to the output HTML file
        pretty: True, for a more human readable HTML string; False, for a smaller sized file.

    Returns:
        HTML string

    """
    # TODO (OM, 20210115): Refactor to remove duplicate code inside 'html_description'?

    FRAME_COLOR = "#ef0707"
    OUTLINE_COLOR = "orange"
    STYLE = """
        .labels {{
          overflow-x: scroll;
          overflow-y: hidden;
          white-space: nowrap;
          -webkit-overflow-scrolling: touch;
        }}
        .task-image {{
          padding: 2px;
          display: inline-block;
        }}
        .task-image img {{
          max-width: {w}px;
          max-height: {h}px;
        }}
        .warn{{
          padding: 2px;
          border: 6px solid {c};
        }}
        .discarded{{
          outline: 3px solid {o};
        }}
        .summary{{
          color: white;
          text-align: center;
          outline: 2px solid #626262;
          outline-style: double;
          background-color: #8c8c8c;
        }}
    """.format(w=width, h=height, c=FRAME_COLOR, o=OUTLINE_COLOR)
    NOTES = ("- Hover on any image to read its best and second best consensus probabilities.<br/>"
             "- When these two probabilities have a difference &le; {t}, the image is framed with "
             "<span style='color:{c}; font-weight:bold'>borders</span> and marked as warning.<br/>"
             "- Images which were discarded by the <em>Pipeline</em> are "
             "<span style='color:{o}; font-weight:bold'>outlined</span>.<br/>"
             "- Scroll horizontally to view all images.").format(t=warn_threshold, c=FRAME_COLOR, o=OUTLINE_COLOR)
    TITLE = "{} Consensus :: {}".format(data.data_src, question.title())

    best = np.argmax(consensus, axis=1)
    labels = np.unique(best)
    label_names = list(data.df[question].cat.categories)

    with document(title=TITLE) as doc:
        with doc.head:
            tags.style(STYLE)
            tags.base(target="_blank")
        with doc.body as body:
            body.add(tags.h1(TITLE, style="color:Tomato; text-align:center"))
            body.add(tags.p(util.raw(NOTES)))
            summary_id = "summary"
            body.add(tags.h1("", id=summary_id, cls="summary"))
            n_warn_total = 0
            n_discard_warn_total = 0
            n_discard_good_total = 0
            for label in labels:
                task_indices = np.where(best == label)[0]
                task_picture_links = data.get_field(task_indices, picture_field, unique=True)
                label_id = "label" + str(label)
                body.add(tags.h2("{} ({}):".format(str(label_names[label]).title(), len(task_indices)), id=label_id))
                n_warn = 0
                n_discard_warn = 0
                n_discard_good = 0
                with body.add(tags.div(cls='labels')):
                    for ix, tpl in enumerate(task_picture_links):
                        l = tags.div(cls="task-image")
                        task_index = task_indices[ix]
                        probabilities = [(p, li) for li, p in enumerate(consensus[task_index, :])]
                        probabilities.sort(key=lambda x: x[0], reverse=True)
                        probs_str = "\n".join("- {0}: {1:.{2}f}".format(
                            label_names[li], p, dec) for p, li in probabilities[:2])
                        img_title = "Task Index: {}\n{}".format(str(task_index), probs_str)
                        is_warning = probabilities[0][0] - probabilities[1][0] <= warn_threshold
                        is_discarded = task_index not in pipeline_task_indices
                        if is_warning:
                            n_warn += 1
                            kwargs = {"cls": "warn"}
                        else:
                            kwargs = {}
                        if is_discarded:
                            if is_warning:
                                n_discard_warn += 1
                            else:
                                n_discard_good += 1
                            kwargs["cls"] = (kwargs.get("cls", "") + " discarded").lstrip()
                        l += tags.a(tags.img(src=tpl, title=img_title, **kwargs), href=tpl)
                    n_discard = n_discard_warn + n_discard_good
                    if n_warn + n_discard > 0:
                        str_label = ""
                        if n_warn > 0:
                            str_label = "<span style='color:{c}'>{n}</span> warning{s}".format(
                                c=FRAME_COLOR, n=str(n_warn), s="s" if n_warn > 1 else "")
                        if n_discard > 0:
                            if str_label:
                                str_label += ", "
                            str_label += "<span style='color:{o}'>{n}</span> discarded".format(
                                o=OUTLINE_COLOR, n=str(n_discard_warn + n_discard_good))
                        label_header = doc.body.getElementById(label_id)
                        label_header.add_raw_string(" ({})".format(str_label))
                n_warn_total += n_warn
                n_discard_warn_total += n_discard_warn
                n_discard_good_total += n_discard_good
            str_summary = ("Consensus Warnings: <span style='color:{c}'>{tw:.1%}</span><br/> "
                           "Discarded Warnings: <span style='color:{o}'>{tdw:.1%}</span>, "
                           "Discarded Good: <span style='color:{o}'>{tdg:.1%}</span>").format(
                c=FRAME_COLOR, o=OUTLINE_COLOR, tw=n_warn_total/consensus.shape[0],
                tdw=0 if n_warn_total == 0 else n_discard_warn_total/n_warn_total,
                tdg=n_discard_good_total/(data.n_tasks - n_warn_total))
            summary_elm = doc.body.getElementById(summary_id)
            summary_elm.add_raw_string(str_summary)

    html_str = doc.render(pretty=pretty, xhtml=True)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(html_str)
        print("HTML for the {} consensus for the question '{}' is saved into file:\n '{}'".format(
            data.data_src, question, os.path.relpath(output_file)))
    return html_str