"""Parsers for the raw sources (run on small inline samples, no network)."""

import pandas as pd

from marketlab.data import fomc, fred, wiki
from marketlab.features import text

WIKI_SAMPLE = """
<div class="current-events-main vevent" id="2020_March_15">
 <div class="current-events-heading"><div class="current-events-title"><span class="summary">March 15, 2020
  <span style="display:none">(<span class="bday dtstart published updated itvstart">2020-03-15</span>)</span></span></div></div>
 <div class="current-events-content description">
  <div class="current-events-content-heading" role="heading">Business and economy</div>
  <ul><li><a href="/wiki/X">COVID-19 pandemic in the United States</a>
     <ul><li>In an emergency move, the Fed drops its benchmark rate a full percentage point.
          <a class="external text" href="http://example.org">(Reuters)</a></li></ul></li></ul>
  <div class="current-events-content-heading" role="heading">Disasters and accidents</div>
  <ul><li>A landslide kills 12 people in the province. <a class="external text" href="http://e.org">(BBC)</a></li></ul>
 </div>
</div>
<div class="current-events-main vevent" id="2008_September_15">
 <div class="current-events-heading"><span class="bday dtstart">2008-09-15</span></div>
 <div class="current-events-content description">
  <ul><li>Lehman Brothers files for Chapter 11 bankruptcy protection.</li></ul>
 </div>
</div>
"""


def test_wiki_parser_extracts_dated_events_with_context_and_without_sources():
    events = wiki.parse_month(WIKI_SAMPLE)
    assert len(events) == 3
    first, second, third = events
    assert first["date"] == pd.Timestamp("2020-03-15") and first["category"] == "Business and economy"
    assert first["text"].startswith("COVID-19 pandemic in the United States:")
    assert "Reuters" not in first["text"] and "BBC" not in second["text"]
    assert second["category"] == "Disasters and accidents"
    assert third["date"] == pd.Timestamp("2008-09-15") and third["category"] == ""


def test_text_topics_tone_and_deaths():
    events = pd.DataFrame(wiki.parse_month(WIKI_SAMPLE))
    scored = text.score_events(events)
    assert scored.loc[0, "central_bank"] == 1.0 and scored.loc[0, "health"] == 1.0 and scored.loc[0, "us"] == 1.0
    assert scored.loc[1, "disaster"] == 1.0 and scored.loc[1, "deaths"] == 12.0
    assert scored.loc[2, "fin_crisis"] == 1.0 and scored.loc[2, "us"] == 0.0
    assert text.death_count("At least 1,200 people are killed in the quake") == 1200.0
    assert text.death_count("A bombing kills at least thirty... no: kills twelve soldiers") == 12.0
    assert text.death_count("The war that began in 2003 killed many") == 0.0
    assert text.death_count("He tells us the vote passed") == 0.0
    daily = text.daily_event_counts(events)
    assert daily.loc["2020-03-15", "n"] == 2.0 and daily.loc["2008-09-15", "n"] == 1.0
    assert daily.loc["2015-01-01", "n"] == 0.0     # days without events are zeros, not gaps


FOMC_CALENDAR = """
<div class="panel panel-default"><div class="panel-heading"><h4><a id="1">2023 FOMC Meetings</a></h4></div>
 <div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>Jan/Feb</strong></div>
  <div class="fomc-meeting__date col-xs-4">31-1</div></div>
 <div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>March</strong></div>
  <div class="fomc-meeting__date col-xs-4">21-22*</div></div>
 <div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5"><strong>August</strong></div>
  <div class="fomc-meeting__date col-xs-4">22 (notation vote)</div></div>
</div>
"""


def test_fomc_parsers():
    rows = fomc.parse_calendar(FOMC_CALENDAR)
    assert [(str(r["date"].date()), r["scheduled"]) for r in rows] == [
        ("2023-02-01", True), ("2023-03-22", True), ("2023-08-22", False)]
    html = "<h5>January 29-30 Meeting - 2008</h5><h5>January 21 Conference Call - 2008</h5>"
    rows = fomc.parse_historical(html, 2008)
    assert [(str(r["date"].date()), r["scheduled"]) for r in rows] == [("2008-01-30", True), ("2008-01-21", False)]


def test_fred_csv_parser_handles_missing_values():
    body = "observation_date,VIXCLS\n2020-01-02,12.5\n2020-01-03,.\n2020-01-06,\n2020-01-07,13.9\n"
    series = fred.parse_series(body, "VIXCLS")
    assert series.index.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-07")]
    assert series.tolist() == [12.5, 13.9]
