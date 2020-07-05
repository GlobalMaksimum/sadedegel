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


class FileManager:
    def __init__(self, folder):

        if not os.path.isdir(folder):
            raise FileNotFoundError("{} is not a directory".format(folder))

        self.root_folder = folder

        # self.root_folder = os.path.join(folder, "labeltool_out")
        self.corpus_folder = pjoin(self.root_folder, "sents")
        self.label_folder = pjoin(self.root_folder, "labels")

        os.makedirs(self.label_folder, exist_ok=True)

        self.corpus_files = glob.glob(pjoin(self.corpus_folder, "*.json"))

        logger.info("|corpus|: {}".format(len(self.corpus_files)))

        self.lengths = [0] * len(self.corpus_files)  # updated by get_ith

        self.idx = -1

    def _get_ith(self, i):
        if not 0 <= i < len(self.corpus_files):
            raise IndexError("Invalid index {}".format(i))

        with open(self.corpus_files[i]) as fp:
            doc = json.load(fp)
            sents = doc['sentences']

        self.lengths[i] = len(sents)

        return sents, [0 for _ in range(self.lengths[i])]

    def combine_jsons(self):
        texts_list = []
        labels_list = []
        for filename in os.listdir(self.sen_folder):
            with open(os.path.join(self.sen_folder, filename), "r") as f:
                sents = json.load(f)
                texts_list.append(sents)

            try:
                with open(os.path.join(self.label_folder, filename), "r") as f:
                    labels_list.append(json.load(f))
            except FileNotFoundError:  # labels do not exist yet
                labels_list.append([0] * len(sents))

        with open(os.path.join(self.root_folder, "texts_combined.json"), "w") as f:
            json.dump(texts_list, f)

        with open(os.path.join(self.root_folder, "labels_combined.json"), "w") as f:
            json.dump(labels_list, f)

    def get_next_sentences(self):
        self.idx += 1

        return self._get_ith(self.idx % len(self.corpus_files))

    def get_prev_sentences(self):
        self.idx -= 1

        return self._get_ith(self.idx % len(self.corpus_files))

    def save_curr_labels(self, labels):
        if not type(labels) == list:
            raise TypeError("labels if of type {}. list expected".format(type(labels)))

        with open(os.path.join(self.label_folder, self.file_list[self.idx] + ".json"), "w") as f:
            json.dump(labels, f)


class SentenceTextPanel(wx.Panel):
    def __init__(self, parent, label, bg=(242, 238, 203)):
        super().__init__(parent)
        self.text = wx.StaticText(parent=self, label=label)
        self.SetBackgroundColour(bg)
        self.selected = False

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.text, flag=wx.EXPAND | wx.ALL, border=5)
        self.Bind(wx.EVT_ENTER_WINDOW, self.OnEnter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnLeave)
        self.text.Bind(wx.EVT_LEFT_UP, self.OnClick)

        self.bg = bg
        self.SetSizer(sizer)

    def OnEnter(self, e):
        if not self.selected:
            self.text.SetForegroundColour((255, 0, 0))

    def OnLeave(self, e):
        if not self.selected:
            self.text.SetForegroundColour((31, 30, 25))

    def OnClick(self, e):
        self.Toggle()

    def Toggle(self):
        if self.selected:
            self.selected = False
            self.text.SetForegroundColour((31, 30, 25))
            self.SetBackgroundColour(self.bg)
        else:
            self.selected = True
            self.text.SetForegroundColour((0, 0, 0))
            self.SetBackgroundColour((242, 238, 203))

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

        self.bg = (215, 208, 141)
        self.SetBackgroundColour(self.bg)

        self.BuildSentences(sentences)
        self.SetSizer(self.sizer)
        self.SetupScrolling()

    def BuildSentences(self, sentences):
        sentence_list, labels = sentences
        self.sizer.Clear(delete_windows=True)
        self.sizer.Layout()
        for i, s in enumerate(sentence_list):
            p = SentenceTextPanel(parent=self, label=s, bg=self.bg)
            p.Wrap(500)
            p.SetFont(self.font)
            self.sizer.Add(p, flag=wx.EXPAND | wx.ALIGN_LEFT)
            if labels[i]:
                p.Toggle()


class BtnPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        # self.SetBackgroundColour((240,240,240))
        self.next_btn = wx.Button(parent=self, label=">")
        self.prev_btn = wx.Button(parent=self, label="<")
        self.summ_btn = wx.ToggleButton(parent=self, label="Sadede gel")
        self.comb_btn = wx.Button(parent=self, label="Combine JSONs")

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

        self.sizer.Add(self.comb_btn, flag=wx.ALL, border=5)
        self.sizer.Add(self.summ_btn, flag=wx.ALL, border=5)
        self.SetSizer(self.sizer)
        self.ResetTexts()

    def ResetTexts(self):
        self.summ_btn.SetValue(False)

        fmgr: FileManager = self.GetParent().file_mgr

        self.order_txt.SetLabel("[{}/{}]".format(fmgr.idx + 1, len(fmgr.corpus_files)))
        self.file_txt.SetLabel(fmgr.corpus_files[fmgr.idx])


class MainFrame(wx.Frame):
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
        self.btn_panel.comb_btn.Bind(wx.EVT_BUTTON, self.OnComb)

        self.text_panel.Bind(wx.EVT_CHAR_HOOK, self.OnNextKey)
        # self.text_panel.Bind(wx.EVT_CHAR_HOOK, self.OnNextKey)
        # self.btn_panel.Bind(wx.EVT_CHAR_HOOK, self.OnNextKey)
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Show()
        self.SetSizeHints((800, 800))
        self.Center()

    def _SaveLabels(self):
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

    def OnComb(self, e):
        self._SaveLabels()
        self.file_mgr.combine_jsons()

    def OnSumm(self, e):
        if self.btn_panel.summ_btn.GetValue():
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
        self.file_mgr.combine_jsons()
        self.Destroy()

        self.Show()


@click.command()
@click.option('--base-dir', default=None, help="base directory")
def main(base_dir):
    if base_dir is None:
        base_dir = pjoin(dirname(__file__), '..', 'dataset', 'sents')

    app = wx.App()

    fmgr = FileManager(base_dir)

    _ = MainFrame(file_mgr=fmgr)
    app.MainLoop()


if __name__ == "__main__":
    main()
