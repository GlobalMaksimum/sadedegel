---
short_name: crew
name: Biz Kimiz?
img: /assets/img/bizkimiz-icon.png
---

<div class="row" id="crew">
    <div class="container">
        {% for author in site.authors%}
            <div class="col-6 col-lg-3">
                <p>
                    <a href="{{author.github}}" target="_blank" rel="nofollow">
                        <img src="{{author.img}}" alt="{{author.name}} {{author.lastName}}"><br>
                        {{author.name}} <br>{{author.lastName}}
                    </a>
                </p>
            </div>
        {% endfor %}
    </div>
</div>
