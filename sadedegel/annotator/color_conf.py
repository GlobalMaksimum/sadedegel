"""
    Colors used by the annotator tool defined here.
"""

sentence_panel = {
    "untoggled":{
        "bg":(215,208,141),
        "fg":(31,30,25),
    },
    "toggled":{
        "bg":(242,238,203),
        "fg":(0,0,0)
    },
    "hovered":{
        "fg":(255,0,0)
    }
}

text_panel = {
    "bg":sentence_panel["untoggled"]["bg"]
}
