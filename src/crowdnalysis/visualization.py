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


def html_description_crossed(consensus: np.ndarray, data: Data, question: str, compare_task_indices: List[int],
                             picture_field: str, lbl_outline: str, lbl_actor: str, included: bool = True,
                             width=120, height=90, dec=3, warn_threshold=.1,
                             output_file: str = None, pretty=True) -> str:
    """Return an HTML string that displays the images of tasks. Outline the tasks which are [in/not in] the given list.

    Optionally, saves the string into an HTML file.

    Args:
        consensus: Consensus probabilities of tasks
        data : Annotation data
        question: Question for annotation
        compare_task_indices: List of task indices of to be compared with the tasks in the `consensus`
        picture_field: column name in the `Data.df` dataframe
        lbl_outline: Label for the tasks whose `included` comparison results `True`
        lbl_actor: The component who generated the `compare_task_indices` (e.g. POLIMI's pipeline)
        included: If True, outlines the `consensus` tasks `ìn` the `compare_task_indices`;
            otherwise, outlines `not in` tasks
        width: max image width
        height: max image height
        dec: number of decimal digits of probabilities to display
        warn_threshold: threshold difference between best and second best consensus probabilities to be marked
            as warning
        output_file: (optional) full path to the output HTML file. If omitted, file is not saved.
        pretty: True, for a more human readable HTML string; False, for a smaller sized file.

    Returns:
        HTML string

    """
    # TODO (OM, 20210115): Refactor to remove duplicated code inside 'html_description'?

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
        .outline{{
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
             "- Otherwise, the image is regarded as 'good' (regardless of whether it is relevant or not).<br/>"
             "- Images which were <b>{m}</b> by <em>{a}</em> are "
             "<span style='color:{o}; font-weight:bold'>outlined</span>.<br/>"
             "- Scroll horizontally to view all images.").format(t=warn_threshold, c=FRAME_COLOR, o=OUTLINE_COLOR,
                                                                 a=lbl_actor, m=lbl_outline.lower())
    TITLE = "{} Consensus :: {}".format(data.data_src, question.title())

    def label_info(n_warn_, n_outline_, msg_action):
        info_ = ""
        if n_warn_ > 0:
            info_ = "<span style='color:{c}'>{n}</span> warning{s}".format(
                c=FRAME_COLOR, n=str(n_warn_), s="s" if n_warn_ > 1 else "")
        if n_outline_ > 0:
            if info_:
                info_ += ", "
            info_ += "<span style='color:{o}'>{n}</span> {m}".format(
                o=OUTLINE_COLOR, n=str(n_outline_), m=msg_action.lower())
        info_ = " ({})".format(info_)
        return info_

    def make_summary(consensus_, data_, n_warn_total_, n_outline_good_total_, n_outline_warn_total_, msg_action):
        str_summary_ = ("Consensus Warnings: <span style='color:{c}'>{tw:.1%}</span><br/>"
                        "{m} Warnings: <span style='color:{o}'>{tdw:.1%}</span>, "
                        "{m} Good: <span style='color:{o}'>{tdg:.1%}</span>").format(
            c=FRAME_COLOR, o=OUTLINE_COLOR, tw=n_warn_total_ / consensus_.shape[0],
            tdw=0 if n_warn_total_ == 0 else n_outline_warn_total_ / n_warn_total_,
            tdg=n_outline_good_total_ / (data_.n_tasks - n_warn_total_),
            m=msg_action.capitalize())
        return str_summary_

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
            n_outline_warn_total = 0
            n_outline_good_total = 0
            for label in labels:
                task_indices = np.where(best == label)[0]
                task_picture_links = data.get_field(task_indices, picture_field, unique=True)
                label_id = "label" + str(label)
                body.add(tags.h2("{} ({}):".format(str(label_names[label]).title(), len(task_indices)), id=label_id))
                n_warn = 0
                n_outline_warn = 0
                n_outline_good = 0
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
                        is_outlined = (included == (task_index in compare_task_indices))
                        if is_warning:
                            n_warn += 1
                            kwargs = {"cls": "warn"}
                        else:
                            kwargs = {}
                        if is_outlined:
                            if is_warning:
                                n_outline_warn += 1
                            else:
                                n_outline_good += 1
                            kwargs["cls"] = (kwargs.get("cls", "") + " outline").lstrip()
                        l += tags.a(tags.img(src=tpl, title=img_title, **kwargs), href=tpl)
                    n_outline = n_outline_warn + n_outline_good
                    if n_warn + n_outline > 0:
                        label_header = doc.body.getElementById(label_id)
                        label_header.add_raw_string(label_info(n_warn, n_outline, lbl_outline))
                n_warn_total += n_warn
                n_outline_warn_total += n_outline_warn
                n_outline_good_total += n_outline_good
            summary_elm = doc.body.getElementById(summary_id)
            summary_elm.add_raw_string(make_summary(consensus, data, n_warn_total,
                                                    n_outline_good_total, n_outline_warn_total, lbl_outline))

    html_str = doc.render(pretty=pretty, xhtml=True)
    if output_file:
        with open(output_file, 'w') as f:
            f.write(html_str)
        print("HTML for the {s} '{q}' consensus vs {a}-{m} tasks is saved into:\n'{f}'".format(
            s=data.data_src, q=question, a=lbl_actor, m=lbl_outline, f=os.path.relpath(output_file)))
    return html_str
