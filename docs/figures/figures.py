from manim_fonts import *
from manim_mobject_svg import *
from manimlib import *
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg


class RenderableArrow(VGroup):
    def __init__(
        self,
        start,
        end,
        buff=0.05,
        stroke_width=1,
        stroke_color=BLACK,
        fill_color=BLACK,
        tip_ratio=0.25,
        **kwargs
    ):
        super().__init__()
        self.line = Line(
            start + buff * RIGHT,
            end - (buff + get_norm(end - start) * tip_ratio / 2) * RIGHT,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            fill_color=fill_color,
            **kwargs
        )
        self.add(self.line)

        self.tip = Triangle(
            fill_color=stroke_color,
            stroke_color=stroke_color,
            fill_opacity=1,
        )
        self.tip.set_width(tip_ratio * self.line.get_length())
        self.tip.move_to(end - buff * RIGHT, aligned_edge=RIGHT)
        self.tip.rotate(self.line.get_angle())
        self.tip.rotate(-90 * DEGREES)
        self.add(self.tip)


class CenterText(VGroup):
    def __init__(self, text, *args, **kwargs):
        super().__init__()
        texts = VGroup(*[Text(i, *args, **kwargs) for i in text.split("\n")])
        texts.arrange(DOWN, center=True)
        self.add(texts)


class Generation(Scene):
    def construct(self):
        rect_space_factor = 5
        titles_text = [
            "Game Description\nGeneration",
            "Implementation\nGeneration",
            "Self-Play\nReinforcement\nLearning",
        ]

        rects = VGroup()
        for i in range(3):
            rect = RoundedRectangle(
                stroke_color=BLACK,
                fill_color=WHITE,
                height=3.5,
                width=4,
            )
            rect.move_to(i * rect_space_factor * RIGHT)
            rects.add(rect)
        rects.center()

        arrows = VGroup()
        for i in range(2):
            arrow = RenderableArrow(
                rects[i].get_right(),
                rects[i + 1].get_left(),
                buff=0.05,
                stroke_width=10,
                stroke_color=BLACK,
                fill_color=BLACK,
            )
            arrows.add(arrow)

        titles = VGroup()
        for i in range(3):
            with RegisterFont("Ubuntu") as fonts:
                title = CenterText(
                    titles_text[i],
                    color=BLACK,
                    font=fonts[0],
                )

            title.scale(0.75)
            title.move_to(rects[i].get_top() + 0.25 * DOWN, aligned_edge=UP)
            titles.add(title)

        group = VGroup(
            rects,
            arrows,
            titles,
        )
        group.to_svg("svg/generation.svg")

        drawing = svg2rlg("svg/generation.svg")
        renderPDF.drawToFile(drawing, "pdf/generation.pdf")

        self.add(rects, arrows, titles)
        self.wait()
        self.embed()
