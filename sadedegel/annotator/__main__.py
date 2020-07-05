import os
import json
import wx
import wx.lib
import wx.lib.scrolledpanel as scrolled

import click
from os.path import dirname
from os.path import join as pjoin
import glob
from loguru import logger

import color_conf

class FileManager:
    """
        Manages loading sentence level tokenized inputs, giving it to GUI in cyclical manner
        and saving the labels.
        "sents" and "labels" folder both contain .json files with same names,
        "sents" containing sentence-level tokenized inputs and "labels" contains labels.

        Constructor args:
            folder: Root folder containing "sents" and "labels" folders.

    """

    def __init__(self, folder):

        if not os.path.isdir(folder):
            raise FileNotFoundError("{} is not a directory".format(folder))

        self.root_folder = folder

        self.corpus_folder = pjoin(self.root_folder, "sents")
        self.label_folder = pjoin(self.root_folder, "labels")
        print(self.label_folder)
        os.makedirs(self.label_folder, exist_ok=True)
        self.corpus_files = glob.glob(pjoin(self.corpus_folder, "*.json"))

        assert len(self.corpus_files) > 0, "Folder is empty!"

        logger.info("|corpus|: {}".format(len(self.corpus_files)))

        self.lengths = [0] * len(self.corpus_files)  # number of sentences per each sentence
        self.idx = -1

    def _get_ith(self, i):
        if not 0 <= i < len(self.corpus_files):
            raise IndexError("Invalid index {}".format(i))

        with open(self.corpus_files[i]) as fp:
            doc = json.load(fp)
            sents = doc['sentences']

        self.lengths[i] = len(sents)

        return sents, [0 for _ in range(self.lengths[i])]


    def get_next_sentences(self):
        self.idx += 1

        return self._get_ith(self.idx % len(self.corpus_files))

    def get_prev_sentences(self):
        self.idx -= 1

        return self._get_ith(self.idx % len(self.corpus_files))

    def save_curr_labels(self, labels):
        if not type(labels) == list:
            raise TypeError("labels if of type {}. list expected".format(type(labels)))

        file_name = os.path.split(self.corpus_files[self.idx])[1]

        with open(pjoin(self.label_folder, file_name), "w") as f:
            json.dump({"labels":labels}, f)


class SentencePanel(wx.Panel):
    """
        A Panel which has a StaticText and which can be toggled on.
    """"

    def __init__(self, parent, label):
        super().__init__(parent)
        self.text = wx.StaticText(parent=self, label=label)
        self.SetBackgroundColour(color_conf.sentence_panel["untoggled"]["bg"])
        self.selected = False

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.text, flag=wx.EXPAND | wx.ALL, border=5)
        self.Bind(wx.EVT_ENTER_WINDOW, self.OnEnter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnLeave)
        self.text.Bind(wx.EVT_LEFT_UP, self.OnClick)

        self.SetSizer(sizer)

    def OnEnter(self, e):
        if not self.selected:
            self.text.SetForegroundColour(color_conf.sentence_panel["hovered"]["fg"])

    def OnLeave(self, e):
        if self.selected:
            self.text.SetForegroundColour(color_conf.sentence_panel["toggled"]["fg"])
        else:
            self.text.SetForegroundColour(color_conf.sentence_panel["untoggled"]["fg"])

    def OnClick(self, e):
        self.Toggle()

    def Toggle(self):
        if self.selected:
            self.selected = False
            self.text.SetForegroundColour(color_conf.sentence_panel["untoggled"]["fg"])
            self.SetBackgroundColour(color_conf.sentence_panel["untoggled"]["bg"])
        else:
            self.selected = True
            self.text.SetForegroundColour(color_conf.sentence_panel["toggled"]["fg"])
            self.SetBackgroundColour(color_conf.sentence_panel["toggled"]["bg"])

    def Wrap(self, wrap):
        self.text.Wrap(wrap)

    def SetFont(self, font):
        self.text.SetFont(font)


class MainTextPanel(scrolled.ScrolledPanel):
    def __init__(self, parent, sentences):
        super().__init__(parent=parent, style=wx.BORDER_SUNKEN)

        self.SetMinSize((600, 600))
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.font = wx.Font(13, wx.MODERN, wx.NORMAL, wx.NORMAL)

        self.SetBackgroundColour(color_conf.text_panel["bg"])

        self.BuildSentences(sentences)
        self.SetSizer(self.sizer)
        self.SetupScrolling()

    def BuildSentences(self, sentences):
        sentence_list, labels = sentences
        self.sizer.Clear(delete_windows=True)
        self.sizer.Layout()
        for i, s in enumerate(sentence_list):
            p = SentencePanel(parent=self, label=s)
            p.Wrap(500)
            p.SetFont(self.font)
            self.sizer.Add(p, flag=wx.EXPAND | wx.ALIGN_LEFT)

            if labels[i]:
                p.Toggle()


class BtnPanel(wx.Panel):
    """
        Bottom panel containing the navigation buttons.
        No logic is done here, but rather the events are bound in MainFrame.
    """

    def __init__(self, parent):
        super().__init__(parent=parent)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        # self.SetBackgroundColour((240,240,240))
        self.next_btn = wx.Button(parent=self, label=">")
        self.prev_btn = wx.Button(parent=self, label="<")
        self.summ_btn = wx.ToggleButton(parent=self, label="Sadede gel")
        # self.comb_btn = wx.Button(parent=self, label="Combine JSONs")

        font = wx.Font(11, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.order_txt = wx.StaticText(parent=self, label="")
        self.order_txt.SetFont(font)

        self.file_txt = wx.StaticText(parent=self, label="")
        self.file_txt.SetFont(font)

        self.sizer.Add(self.prev_btn, flag=wx.ALL | wx.EXPAND, border=5)
        self.sizer.Add(self.next_btn, flag=wx.ALL | wx.EXPAND, border=5)
        self.sizer.AddStretchSpacer()

        self.sizer.Add(self.order_txt, flag=wx.ALL | wx.ALIGN_CENTRE, border=5)
        self.sizer.Add(self.file_txt, flag=wx.ALL | wx.ALIGN_CENTER, border=5)
        self.sizer.AddStretchSpacer()

        # self.sizer.Add(self.comb_btn, flag=wx.ALL, border=5)
        self.sizer.Add(self.summ_btn, flag=wx.ALL, border=5)
        self.SetSizer(self.sizer)
        self.ResetTexts()

    def ResetTexts(self):
        """
            Change the index and filename shown.
        """

        self.summ_btn.SetValue(False)

        fmgr: FileManager = self.GetParent().file_mgr

        self.order_txt.SetLabel("[{}/{}]".format(fmgr.idx + 1, len(fmgr.corpus_files)))
        file_name = os.path.split(fmgr.corpus_files[fmgr.idx])[1]
        self.file_txt.SetLabel(file_name)


class MainFrame(wx.Frame):
    """
        Main container frame.
        All logic is bound to GUI layer here.

        Constructor args:
            file_mgr: FileManager instance which handles corpus files.
    """

    def __init__(self, file_mgr: FileManager):
        super().__init__(parent=None, title="Sadedegel Annotation Tool")
        self.file_mgr = file_mgr
        self.SetTitle("Sadedegel Annotation Tool - ({})".format(self.file_mgr.root_folder))
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        # sizer = wx.FlexGridSizer(cols=1)
        # main_panel = wx.Panel(self)
        self.text_panel = MainTextPanel(parent=self, sentences=self.file_mgr.get_next_sentences())
        self.btn_panel = BtnPanel(parent=self)

        self.sizer.Add(self.text_panel, flag=wx.ALL | wx.EXPAND, proportion=1)
        self.sizer.Add(self.btn_panel, flag=wx.ALIGN_RIGHT | wx.EXPAND, proportion=0)
        # btn_panel.FitInside(s)
        self.SetSizer(self.sizer)
        self.sizer.Layout()

        self.btn_panel.next_btn.Bind(wx.EVT_BUTTON, self.OnNext)
        self.btn_panel.prev_btn.Bind(wx.EVT_BUTTON, self.OnPrev)
        self.btn_panel.summ_btn.Bind(wx.EVT_TOGGLEBUTTON, self.OnSumm)

        self.text_panel.Bind(wx.EVT_CHAR_HOOK, self.OnNextKey)
        # self.text_panel.Bind(wx.EVT_CHAR_HOOK, self.OnNextKey)
        # self.btn_panel.Bind(wx.EVT_CHAR_HOOK, self.OnNextKey)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Show()
        self.SetSizeHints((800, 800))
        self.Center()

    def _SaveLabels(self):
        """
            Creates a list of 1's and 0's denoting whether the sentence is important or not,
            and passes it on to the FileManager for saving.
        """

        labels = [0] * self.file_mgr.lengths[self.file_mgr.idx]

        for i, t in enumerate(self.text_panel.sizer.GetChildren()):
            t = t.GetWindow()
            if t.selected:
                labels[i] = 1

        self.file_mgr.save_curr_labels(labels)

    def _OnPageChange(self):
        self.btn_panel.ResetTexts()
        self.sizer.Layout()
        # self.OnSumm

    def OnNextKey(self, e):
        if e.GetUnicodeKey() == 32:
            self.OnNext(e)
            self.SetFocus()

    def OnNext(self, e):
        self._SaveLabels()
        self.text_panel.BuildSentences(self.file_mgr.get_next_sentences())
        self._OnPageChange()

    def OnPrev(self, e):
        self._SaveLabels()
        self.text_panel.BuildSentences(self.file_mgr.get_prev_sentences())
        self._OnPageChange()

    def OnSumm(self, e):
        if self.btn_panel.summ_btn.GetValue(): # when summarize button is toggled
            for i, t in enumerate(self.text_panel.sizer.GetChildren()):
                t = t.GetWindow()

                if not t.selected:
                    t.Hide()

            # self.text_panel.Layout()
        else:
            for i, t in enumerate(self.text_panel.sizer.GetChildren()):
                t = t.GetWindow()

                if not t.selected:
                    t.Show()

        self.text_panel.ScrollChildIntoView(self.text_panel)
        self.Layout()
        self.text_panel.Layout()

    def OnClose(self, e):
        self._SaveLabels()
        self.Destroy()


@click.command()
@click.option('--base-dir', default=None, help="base directory")
def main(base_dir):
    if base_dir is None:
        base_dir = pjoin(dirname(__file__), os.pardir, "dataset", "preprocessed")
        base_dir = os.path.abspath(base_dir)
    app = wx.App()

    fmgr = FileManager(base_dir)

    _ = MainFrame(file_mgr=fmgr)
    app.MainLoop()


if __name__ == "__main__":
    main()
