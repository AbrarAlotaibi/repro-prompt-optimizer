"""
Customizable Rich layout for the monitoring CLI.
Edit this file and enable hot reload to update the layout without restarting.
"""

from rich.layout import Layout


def create_layout(display):
    """Create the layout using the display's panel helpers."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    layout["body"].split_row(
        Layout(name="main", ratio=2),
        Layout(name="sidebar", ratio=1),
    )

    layout["main"].split_column(
        Layout(name="progress", size=12),
        Layout(name="metrics"),
    )

    layout["header"].update(display._create_header())
    layout["progress"].update(display._create_progress_panel())
    layout["metrics"].update(display._create_metrics_panel())
    layout["sidebar"].update(display._create_sidebar())
    layout["footer"].update(display._create_footer())

    return layout
