"""Smoke tests for the demo CLI: every topic runs and prints something."""

import pytest

from calccode import demo


@pytest.mark.parametrize("topic", sorted(demo.TOPICS))
def test_demo_topic_prints_output(topic, capsys):
    assert demo.main([topic]) == 0
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) >= 3


def test_demo_no_args_lists_topics(capsys):
    assert demo.main([]) == 0
    out = capsys.readouterr().out
    for topic in demo.TOPICS:
        assert topic in out


def test_demo_unknown_topic_fails(capsys):
    assert demo.main(["bogus"]) == 1
    assert "unknown topic" in capsys.readouterr().out
