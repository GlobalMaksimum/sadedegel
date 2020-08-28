from .context import Rouge1Summarizer 
from .context import KMeansSummarizer, AutoKMeansSummarizer, DecomposedKMeansSummarizer 
from .context import RandomSummarizer, PositionSummarizer, LengthSummarizer, BandSummarizer

def test_baseline_tags():

    rand = RandomSummarizer()
    pos = PositionSummarizer()
    length = LengthSummarizer()
    band = BandSummarizer()
    
    assert ("baseline" in rand) == True 
    assert ("baseline" in pos) == True
    assert ("baseline" in length) == True 
    assert ("baseline" in band) == True

def test_cluster_tags():

    km = KMeansSummarizer()
    autokm = AutoKMeansSummarizer()
    decomkm = DecomposedKMeansSummarizer()
    
    assert ("cluster" in km) == True 
    assert ("cluster" in autokm) == True
    assert ("cluster" in decomkm) == True 

def test_ss_tags():

    rouge1 = Rouge1Summarizer()
    
    assert ("selfsupervised" in rouge1) == True 
