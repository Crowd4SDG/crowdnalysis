from html.parser import HTMLParser
from unittest.mock import mock_open, patch
# TODO (OM, 20210702): Replace unittest with pytest, if a solid way of mocking `builtins.open` is found for the latter.)

import pandas as pd
import pytest

from .conftest import TEST
from ..factory import Factory
from ..visualization import csv_description, html_description, html_description_crossed


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


def check_valid_html(html: str) -> HTMLParserForTest:
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


def test_html_description(fixt_consensus, fixt_data):

    mocked_file_path = "mock/file/path/spam.html"
    with patch("builtins.open", mock_open()) as mocked_file:
        html = html_description(consensus=fixt_consensus, data=fixt_data, question=question,
                                picture_field=picture_field,
                                output_file=mocked_file_path)

        # assert if opened file on write mode 'w'
        mocked_file.assert_called_once_with(mocked_file_path, 'w')

        # assert if the html content was written in file
        mocked_file().write.assert_called_once_with(html)
    check_valid_html(html)


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

    # Assert generated HTML is valid
    parser = check_valid_html(html)

    html2 = html_description(consensus=fixt_consensus, data=fixt_data, question=question, picture_field=picture_field)
    parser2 = HTMLParserForTest()
    parser2.feed(html2)

    assert(len(parser.start_tags) > len(parser2.start_tags))


def test_csv_description(fixt_consensus, fixt_data):
    mocked_file_path = "mock/file/path/spam.csv"
    with patch.object(pd.DataFrame, "to_csv") as mocked_file:
        csv = csv_description(consensus=fixt_consensus, data=fixt_data, question=question,
                              picture_field=picture_field, output_file=mocked_file_path)

        # assert if the html content was written in file
        mocked_file.assert_called_with(mocked_file_path)



