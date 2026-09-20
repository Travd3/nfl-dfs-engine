"""Pure-stdlib tests for FanDuel template export."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fanduel_export


def sample_lineup():
    players = []
    for pos, ids in {
        "QB": ["q"], "RB": ["r1", "r2", "r3"], "WR": ["w1", "w2", "w3"],
        "TE": ["t"], "DEF": ["d"],
    }.items():
        for i, fid in enumerate(ids):
            players.append({"position": pos, "fanduel_id": fid,
                            "projection": 20.0 - i})
    return {"rank": 1, "flex_position": "RB", "players": players}


def test_export_preserves_private_metadata_and_fills_nine_slots(tmp_path):
    template = tmp_path / "template.csv"
    out = tmp_path / "filled.csv"
    template.write_text(
        "entry_id,contest_id,contest_name,entry_fee,QB,RB,RB,WR,WR,WR,TE,FLEX,DEF,,Instructions\n"
        "PRIVATE_ENTRY,PRIVATE_CONTEST,Example,0,,,,,,,,,,,,instruction\n"
        ",,,,,,,,,,,,,,,\n"
        ",,,,,,,,,,,,,,,Player ID + Player Name,Id,Position\n",
        encoding="utf-8",
    )
    fanduel_export.fill_template(str(template), sample_lineup(), str(out))

    rows = list(csv.reader(open(out, newline="", encoding="utf-8")))
    header, entry = rows[0], rows[1]
    assert entry[0] == "PRIVATE_ENTRY"
    assert entry[1] == "PRIVATE_CONTEST"

    idx = [header.index("QB"), *[i for i,c in enumerate(header) if c == "RB"],
           *[i for i,c in enumerate(header) if c == "WR"],
           header.index("TE"), header.index("FLEX"), header.index("DEF")]
    chosen = [entry[i] for i in idx]
    assert len(chosen) == 9
    assert len(set(chosen)) == 9
    assert set(chosen) == {"q","r1","r2","r3","w1","w2","w3","t","d"}
