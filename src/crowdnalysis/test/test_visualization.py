from html.parser import HTMLParser
from unittest.mock import mock_open, patch
# TODO (OM, 20210702): Replace unittest with pytest, if a solid way of mocking `builtins.open` is found for the latter.

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .conftest import TEST
from ..factory import Factory
from ..visualization import consensus_as_df, csv_description, html_description, html_description_crossed, plot_confusion


question = TEST.QUESTIONS[0]
picture_field = TEST.EXTRA_COL


class HTMLParserForTest(HTMLParser):

    def __init__(self):
        self.start_tags = []
        self.end_tags = []
        super().__init__()

    def handle_starttag(self, tag, attrs):
        self.start_tags.append(tag)

    def handle_endtag(self, tag):
        self.end_tags.append(tag)


def assert_valid_html(html: str) -> HTMLParserForTest:
    """Asserts if a given string is valid html"""
    parser = HTMLParserForTest()
    parser.feed(html)

    assert (parser.start_tags[0].lower() == "html")
    assert (parser.end_tags[-1].lower() == "html")
    assert (sorted(parser.start_tags) == sorted(parser.end_tags))  # all tags are closed, valid XHTML

    return parser


@pytest.fixture(scope="module")
def fixt_consensus(fixt_data):
    model = Factory.make("MajorityVoting")
    consensus, _ = model.fit_and_compute_consensus(fixt_data.get_dcp(question))
    return consensus


def test_consensus_as_df(fixt_consensus, fixt_data):
    df = consensus_as_df(fixt_data, question, fixt_consensus)
    assert type(df) == pd.DataFrame
    assert np.array_equal(df.index, fixt_data.valid_task_ids(question))
    assert np.array_equal(df.columns, fixt_data.get_class_ids(question))


def test_plot_confusion(monkeypatch, fixt_data):
    model = Factory.make("DawidSkene")
    _, params = model.fit_and_compute_consensus(fixt_data.get_dcp(question))

    fig_saved = False

    def mock_savefig(*args, **kwargs):
        nonlocal fig_saved
        fig_saved = True

    def mock_plt(*args, **kwargs):
        pass

    # Mock used Matplotlib objects' methods
    monkeypatch.setattr(Figure, "savefig", mock_savefig)
    monkeypatch.setattr(plt, "show", mock_plt)
    monkeypatch.setattr(plt, "close", mock_plt)
    # Plot the matrix
    classes_ = fixt_data.get_class_ids(question)
    labels = fixt_data.get_categories()[question].categories.to_list()
    mocked_file_path = "mock/file/path/mock.jpg"
    ax = plot_confusion(conf_mtx=params.pi[0], classes=classes_, labels=labels, filename=mocked_file_path)
    # Assert x and y axes values correspond to their given values
    x_plot = [text_.get_text() for text_ in ax.get_xticklabels()]
    y_plot = [text_.get_text() for text_ in ax.get_yticklabels()]
    assert np.array_equal(x_plot, labels)
    assert np.array_equal(y_plot, classes_)
    # Assert the figure is saved
    assert fig_saved


def test_html_description(fixt_consensus, fixt_data):
    mocked_file_path = "mock/file/path/spam.html"
    with patch("builtins.open", mock_open()) as mocked_file:
        html = html_description(consensus=fixt_consensus, data=fixt_data, question=question,
                                picture_field=picture_field, output_file=mocked_file_path)
        # assert if opened file on write mode 'w'
        mocked_file.assert_called_once_with(mocked_file_path, 'w')
        # assert if the html content was written in file
        mocked_file().write.assert_called_once_with(html)
    # assert generated HTML is valid
    assert_valid_html(html)


def test_html_description_crossed(fixt_consensus, fixt_data):
    compare_task_indices = list(range(len(TEST.TASK_IDS)))[1:]
    mocked_file_path = "mock/file/path/spam_crossed.html"
    with patch("builtins.open", mock_open()) as mocked_file:
        html = html_description_crossed(consensus=fixt_consensus, data=fixt_data, question=question,
                                        picture_field=picture_field, compare_task_indices=compare_task_indices,
                                        lbl_outline="spam", lbl_actor="eggs", output_file=mocked_file_path)
        # assert if opened file on write mode 'w'
        mocked_file.assert_called_once_with(mocked_file_path, 'w')
        # assert if the html content was written in file
        mocked_file().write.assert_called_once_with(html)
    # assert generated HTML is valid
    parser = assert_valid_html(html)

    html2 = html_description(consensus=fixt_consensus, data=fixt_data, question=question, picture_field=picture_field)
    parser2 = HTMLParserForTest()
    parser2.feed(html2)
    # assert `html` has more tags than `html2` due to `compare_task_indices`
    assert(len(parser.start_tags) > len(parser2.start_tags))


def test_csv_description(fixt_consensus, fixt_data):
    mocked_file_path = "mock/file/path/spam.csv"
    with patch.object(pd.DataFrame, "to_csv") as mocked_file:
        csv = csv_description(consensus=fixt_consensus, data=fixt_data, question=question,
                              picture_field=picture_field, output_file=mocked_file_path)
        # assert if the html content was written in file
        mocked_file.assert_called_with(mocked_file_path)
