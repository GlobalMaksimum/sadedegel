---
short_name: how
div_id: nasil
title: Nasıl Kullanılır?
parent: false
parent_name: what
---

<h1><strong>Sadedegel</strong> Nasıl Kullanılır?</h1>
<h3><strong>Kurulum</strong></h3>
<ul>
    <li><strong>İşletim Sistemi: </strong>macOS / OS X · Linux · Windows (Cygwin, MinGW, Visual
        Studio)
    </li>
    <li><strong>Python: </strong>3.6+ versiyonu (sadece 64bit)</li>
    <li><strong>Paket Yöneticisi: </strong>pip</li>
</ul>
<pre class="code-view"><code class="javascript"><span class="hljs-function"><span
    class="hljs-keyword">$ pip</span></span> <span class="hljs-keyword">install</span> <span
    class="hljs-title">sadedegel</span></code></pre>
<br />
<h3><strong>Nasıl Kullanılır?</strong></h3>
<div class="row">
    <div class="col-md-4">
        <br/>
        <p>Kurulum yapıldıktan sonra sadedegel.load() ile kullanabilirsiniz.</p>
</div>
<div class="col-md-8">
    <pre class="code-view">
                        <code class="python">import sadedegel
from <span class="hljs-variable">sadedegel.dataset</span> import load_sentence_corpus, load_raw_corpus
<span class="hljs-variable">nlp</span> = <span class="hljs-variable">sadedegel.load()</span>
<span class="hljs-variable">tokenized</span> = <span class="hljs-variable">load_sentence_corpus()</span>
<span class="hljs-variable">raw</span> = <span class="hljs-variable">load_raw_corpus()</span>
<span class="hljs-variable">summary</span> = <span class="hljs-variable">nlp(raw[0])</span>
<span class="hljs-variable">summary</span> = <span class="hljs-variable">nlp(tokenized[0], sentence_tokenizer=False)</span>
</code>
</pre>
</div>

                    </div>
