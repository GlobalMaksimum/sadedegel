var sidebarWidth = function () {
    if (window.matchMedia("(min-width: 768px)").matches) {
        setTimeout(function () {
            $("#side-bar").css({
                width: $("#side-bar").parent().width(),
            });
        }, 200);
    } else {
        $("#side-bar").removeAttr("style");
    }
    loadjscssfile(
        "https://cdnjs.cloudflare.com/ajax/libs/malihu-custom-scrollbar-plugin/3.1.5/jquery.mCustomScrollbar.min.css",
        "css"
    );
    if (window.matchMedia("(min-width: 1024px)").matches) {
        $("#side-bar").mCustomScrollbar({
            theme: "light",
        });
        $(".on-side").css("height", window.innerHeight);
    } else {
        $("#side-bar").mCustomScrollbar("destroy");
        $(".on-side").removeAttr("style");
    }
};
var detailLoaded = function () {
    setTimeout(function () {
        sidebarWidth();
    }, 200);
};
detailLoaded();
window.addEventListener("orientationchange", detailLoaded);
window.addEventListener("resize", detailLoaded);

var lastAnchor = null;
function openPage(targetEl) {
    $("#content-container")
        .find(".info:visible")
        .fadeOut(350, function () {
            $("#content-container").find(targetEl).fadeIn(350);
        });
    if (lastAnchor == null) {
        $("a[href$='#nedir']").removeClass("active");
        lastAnchor = $("a[href$='${targetEl}']");
        lastAnchor.toggleClass("active");
    }
}

$(document).ready(function () {
    var pages = [
        "#nedir",
        "#neden",
        "#nasil",
        "#mimari",
        "#kurulum",
        "#ekip",
        "#ChromeEklentisi",
        "#ornekler",
        "#VeriToplamaAraci",
        "#VeriEtiketlemeAraci",
    ];
    for (var word = 0; word < pages.length; word++) {
        if (window.location.href.indexOf(pages[word]) > -1) {
            switch (pages[word]) {
                case "#nedir":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#neden":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#nasil":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#mimari":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#kurulum":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#ekip":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#ChromeEklentisi":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#ornekler":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#VeriToplamaAraci":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                case "#VeriEtiketlemeAraci":
                    var targetEl = pages[word];
                    openPage(targetEl);
                    break;
                default:
                    var targetEl = "#nedir";
                    openPage(targetEl);
                    break;
            }
        }
    }
    $(".list-group-item").click(function (e) {
        e.preventDefault();
        var targetEl = $(this).attr("href");
        openPage(targetEl);
        lastAnchor.toggleClass("active");
        $(this).toggleClass("active");
        lastAnchor = $(this);
    });
});
