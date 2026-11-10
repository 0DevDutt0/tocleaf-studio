"""Generate the original test PDF used by the structure, dataset and retrieval tests.

The text below was written for this project. It is not taken from any published book.
Run with:  uv run python tests/fixtures/make_fixture_pdf.py

Structure of the fixture (outline levels in brackets):
  Chapter 1 Soil Water Foundations  [0, parent with intro text]
    1.1 Water Sources                  [1, leaf]
    1.2 Soil Moisture                  [1, parent]
      1.2.1 Field Capacity             [2, leaf]
      1.2.2 Permanent Wilting Point    [2, leaf]
    1.3 Drainage                       [1, leaf]
  Chapter 2 Measuring Crop Water Needs [0, parent]
    2.1 Evapotranspiration             [1, leaf, spans two pages]
    2.2 Crop Coefficients              [1, parent]
      2.2.1 Initial Stage              [2, leaf]
      2.2.2 Mid-Season Stage           [2, leaf]
      2.2.3 Late Season Stage          [2, leaf]
    2.3 Sensor Calibration             [1, leaf]
  Chapter 3 Scheduling and Economics   [0, parent]
    3.1 Irrigation Scheduling          [1, leaf]
    3.2 Cost Accounting                [1, parent]
      3.2.1 Pumping Costs              [2, leaf]
      3.2.2 Labour Costs               [2, leaf]
    3.3 Water Pricing                  [1, leaf]
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

BOOK_TITLE = "Practical Irrigation Scheduling"
PAGE_W, PAGE_H = A4
MARGIN = 60
BODY_SIZE = 11
LEADING = 15

# (title, outline level, body paragraphs). Headings are written exactly as they appear in the outline.
CONTENT: list[tuple[str, int, list[str]]] = [
    (
        "Chapter 1: Soil Water Foundations",
        0,
        [
            "This chapter explains how soil holds water and why that matters when an irrigation schedule is planned.",
        ],
    ),
    (
        "1.1 Water Sources",
        1,
        [
            "Irrigation water comes from surface reservoirs, canals and groundwater wells. Each source has a different "
            "salinity level and a different seasonal availability. Growers should test water salinity before the season "
            "begins, because salinity above 1.5 dS/m reduces yields in many vegetables.",
        ],
    ),
    (
        "1.2 Soil Moisture",
        1,
        [
            "Soil moisture is the water held in the root zone. Agronomists describe it with three reference points: "
            "saturation, field capacity and the permanent wilting point. The water between field capacity and the "
            "permanent wilting point is the plant-available water.",
        ],
    ),
    (
        "1.2.1 Field Capacity",
        2,
        [
            "Field capacity is the amount of water a soil holds after free drainage has largely stopped, usually one to "
            "three days after a thorough irrigation. For a sandy loam it is roughly 12 percent by volume. Above field "
            "capacity, extra water drains away and does not stay available to the roots.",
        ],
    ),
    (
        "1.2.2 Permanent Wilting Point",
        2,
        [
            "The permanent wilting point is the soil moisture level at which plants can no longer extract water. For a "
            "sandy loam it is roughly 5 percent by volume, so the plant-available water is about 7 percent. Plants that "
            "reach this point do not recover when water is added unless the soil is irrigated soon enough.",
        ],
    ),
    (
        "1.3 Drainage",
        1,
        [
            "Drainage removes water that the soil cannot hold. Poorly drained fields need tile drains or raised beds. "
            "Waterlogging for more than 48 hours usually damages the roots of cereal crops, and the damage shows as "
            "yellowing leaves in the following week.",
        ],
    ),
    (
        "Chapter 2: Measuring Crop Water Needs",
        0,
        [
            "Crop water need is estimated from weather records and from the growth stage of the crop. The estimates in this "
            "chapter are planning values and should be checked against local measurements.",
        ],
    ),
    (
        "2.1 Evapotranspiration",
        1,
        [
            "Evapotranspiration combines two losses of water to the air: evaporation from the soil surface and transpiration "
            "through the leaves of the crop. The reference evapotranspiration, usually written ET0, describes the loss from "
            "a well-watered grass surface under the local weather. Weather stations report ET0 each day from temperature, "
            "humidity, wind speed and solar radiation.",
            "To convert ET0 into the water need of a specific crop, the reference value is multiplied by a crop coefficient. "
            "The product is called crop evapotranspiration, and it is the amount that an irrigation schedule has to replace. "
            "On a hot and windy day the same crop can need twice the water it needs on a cool and humid day, even when both "
            "days fall in the same growth stage.",
            "Measured data are more reliable than estimates whenever they are available. A weighing lysimeter records the "
            "water loss of a small block of soil and plants every hour. Few farms own one, so most growers rely on the "
            "weather station nearest to the field. When the station is far away, the grower should compare its readings "
            "with the rainfall recorded in the field for at least two weeks before trusting the estimate.",
            "Rainfall also needs careful treatment. Light showers of less than 3 millimetres often wet only the top layer of "
            "soil and evaporate within a day, so they should not be subtracted from the irrigation requirement. Heavy rainfall "
            "that infiltrates into the root zone is counted in full, but only up to the field capacity of the soil. Any "
            "remainder drains below the root zone and is lost from the account.",
            "Records should be kept for every field. The record lists the daily ET0, the rainfall, the irrigation applied "
            "and the estimated soil water balance. Comparing the balance with the soil sensor readings each week shows "
            "whether the estimates are drifting. A drift of more than 10 millimetres over a fortnight is a reason to check "
            "the sensors and the rainfall gauge before changing the schedule.",
        ],
    ),
    (
        "2.2 Crop Coefficients",
        1,
        [
            "A crop coefficient links the reference evapotranspiration to the crop's own water use. The coefficient changes "
            "through the season, so growers use one value for each growth stage.",
        ],
    ),
    (
        "2.2.1 Initial Stage",
        2,
        [
            "During the initial stage, from planting until about 10 percent ground cover, the crop coefficient is about 0.3 "
            "to 0.4. Evaporation from the soil surface dominates the water loss in this stage, so light and frequent "
            "irrigation is often better than a large application.",
        ],
    ),
    (
        "2.2.2 Mid-Season Stage",
        2,
        [
            "At full canopy the crop coefficient reaches its peak, typically 1.05 to 1.20 for maize. Transpiration from the "
            "canopy becomes the main water loss, and the crop is most sensitive to water shortage during flowering.",
        ],
    ),
    (
        "2.2.3 Late Season Stage",
        2,
        [
            "As the leaves age and senesce, the coefficient falls. Fields that are close to harvest need a coefficient of "
            "about 0.4 to 0.6, and irrigation can be reduced to keep the soil from becoming waterlogged before harvest.",
        ],
    ),
    (
        "2.3 Sensor Calibration",
        1,
        [
            "Soil moisture sensors must be calibrated against gravimetric samples taken from the same depth. Calibration "
            "should be repeated for each soil texture class, because sensor readings drift with salinity and temperature.",
        ],
    ),
    (
        "Chapter 3: Scheduling and Economics",
        0,
        [
            "Scheduling decides when to irrigate and how much water to apply. Costs decide which system a grower can afford "
            "to run for the whole season.",
        ],
    ),
    (
        "3.1 Irrigation Scheduling",
        1,
        [
            "A common scheduling rule irrigates when the root zone has lost about half of its plant-available water. The "
            "depth of water to apply equals the deficit up to field capacity, divided by the application efficiency of the "
            "system. Drip systems usually reach 90 percent efficiency, while furrow systems often reach only 60 percent.",
        ],
    ),
    (
        "3.2 Cost Accounting",
        1,
        [
            "The cost of irrigation is split into energy, labour and water charges. Each part responds to a different decision, "
            "so the accounts are kept separately.",
        ],
    ),
    (
        "3.2.1 Pumping Costs",
        2,
        [
            "Pumping cost depends on the lift height, the efficiency of the pump and the electricity tariff. A deeper well "
            "raises the energy used per cubic metre of water roughly in proportion to the lift height, so a pump that is "
            "worn or badly matched to the well can double the energy bill.",
        ],
    ),
    (
        "3.2.2 Labour Costs",
        2,
        [
            "Labour costs include manual valve checks and filter cleaning. A drip system with automated controllers usually "
            "reduces weekly labour hours compared with furrow irrigation, although the controllers need technical servicing.",
        ],
    ),
    (
        "3.3 Water Pricing",
        1,
        [
            "Many districts charge by the volume of water delivered. Tiered pricing raises the unit price above a seasonal "
            "allocation, which discourages over-irrigation in dry years.",
        ],
    ),
]


class _Layout:
    """Tracks the current page so every page gets the running header and page number."""

    def __init__(self, pdf: canvas.Canvas):
        self.pdf = pdf
        self.page_number = 1
        self._decorate()

    def _decorate(self) -> None:
        self.pdf.setFont("Helvetica-Oblique", 9)
        self.pdf.drawString(MARGIN, PAGE_H - 40, BOOK_TITLE)
        self.pdf.setFont("Helvetica", 9)
        self.pdf.drawCentredString(PAGE_W / 2, 30, str(self.page_number))

    def new_page(self) -> float:
        self.pdf.showPage()
        self.page_number += 1
        self._decorate()
        return PAGE_H - MARGIN - 10


def _draw_wrapped(layout: _Layout, text: str, y: float, font: str, size: int, width: float) -> float:
    lines = simpleSplit(text, font, size, width)
    for line in lines:
        if y < MARGIN:
            y = layout.new_page()
        layout.pdf.setFont(font, size)
        layout.pdf.drawString(MARGIN, y, line)
        y -= LEADING
    return y


def build(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)
    pdf.setTitle(BOOK_TITLE)
    pdf.setAuthor("Tocleaf test fixture")
    pdf.setSubject("Original fixture text for structure and retrieval tests")

    layout = _Layout(pdf)
    width = PAGE_W - 2 * MARGIN
    y = PAGE_H - MARGIN - 10

    for index, (title, level, paragraphs) in enumerate(CONTENT):
        key = f"heading-{index:02d}"
        pdf.bookmarkPage(key)
        pdf.addOutlineEntry(title, key, level=level, closed=False)
        heading_size = 16 if level == 0 else (13 if level == 1 else 12)
        y -= 6
        if y < MARGIN + 3 * LEADING:
            y = layout.new_page()
        pdf.setFont("Helvetica-Bold", heading_size)
        pdf.drawString(MARGIN, y, title)
        y -= heading_size + 8
        for paragraph in paragraphs:
            y = _draw_wrapped(layout, paragraph, y, "Helvetica", BODY_SIZE, width)
            y -= 6
        y -= 8

    pdf.showPage()
    pdf.save()
    return output_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("irrigation_handbook.pdf")
    print(build(target))
