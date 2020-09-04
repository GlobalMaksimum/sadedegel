---
short_name: tagging
div_id: VeriEtiketlemeAraci
title: Veri Etiketleme Araci
---

# **Sadedegel** Veri Etiketleme Aracı

Sadedegel projesi kapsamında geliştirdiğimiz veri etiketleme aracını kullanarak, extraction based özetleme tekniği ile özetlenmiş veri setlerini hızlıca oluşturabilir ve makine öğrenmesi projelerinizde kullanabilirsiniz.

[Electronjs](https://www.electronjs.org) tabanlı, cross-platform ve açık kaynak kodlu bir uygulamadır. Detaylara [Github](https://github.com/GlobalMaksimum/sadedegel-annotator) üzerinden ulaşabilirsiniz.

<br/>
### **Etiketleme Algoritması**

Sadedegel Veri Etiketleme aracı, her turda metnin anlamına en az katkı sağlayan cümlelerin elenmesi ile ilerler. Metin en kısa haline ulaşana kadar bu işlem devam eder. Cümlelerin hangi turda elendiği bilgisi, cümlenin metin içindeki önemini gösteren bir etiket oluşturur.

<div class="row">
                        <div class="col-md-4">
                            <br>
                            <video width="250" height="330" controls="" autoplay="">
                                <source src="/sadedegel/assets/img/annotator.mp4" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                        <div class="col-md-8">
                            <h3><strong>Kurulum</strong></h3>
                            <pre class="code-view"><code class="javascript hljs"><table class="hljs-ln"><tbody><tr><td class="hljs-ln-line hljs-ln-numbers" data-line-number="1"><div class="hljs-ln-n" data-line-number="1"></div></td><td class="hljs-ln-line hljs-ln-code" data-line-number="1"><span class="hljs-function"><span class="hljs-keyword">$ git clone https:<span class="">//github.com/GlobalMaksimum/sadedegel-annotator.git</span></span></span></td></tr><tr><td class="hljs-ln-line hljs-ln-numbers" data-line-number="2"><div class="hljs-ln-n" data-line-number="2"></div></td><td class="hljs-ln-line hljs-ln-code" data-line-number="2"><span class="hljs-function"><span class="hljs-keyword">$ cd</span> <span class="hljs-title">sadedegel-annotator</span></span></td></tr><tr><td class="hljs-ln-line hljs-ln-numbers" data-line-number="3"><div class="hljs-ln-n" data-line-number="3"></div></td><td class="hljs-ln-line hljs-ln-code" data-line-number="3"><span class="hljs-function"><span class="hljs-keyword">$ npm</span></span> <span class="hljs-keyword">install</span> <span class="hljs-title">sadedegel</span></td></tr></tbody></table></code></pre>
                            <br>
                            <h3><strong>Başlangıç</strong></h3>
                            <pre class="code-view"><code class="javascript hljs"><span class="hljs-function"><span class="hljs-keyword">$ npm start</span>     </span></code></pre>
                        </div>
                    </div>
