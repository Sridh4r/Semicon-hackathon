"""Cell model definitions for DRAM pattern generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DRAMConfig:
    """Parameters for a simplified periodic DRAM structure."""

    image_width: int
    image_height: int
    rows: int
    columns: int
    row_pitch: int
    column_pitch: int
    wordline_thickness: int
    bitline_thickness: int
    contact_size: int
    background_intensity: int = 16
    wordline_intensity: int = 170
    bitline_intensity: int = 170
    contact_intensity: int = 255

    def validate(self) -> None:
        checks = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "rows": self.rows,
            "columns": self.columns,
            "row_pitch": self.row_pitch,
            "column_pitch": self.column_pitch,
            "wordline_thickness": self.wordline_thickness,
            "bitline_thickness": self.bitline_thickness,
            "contact_size": self.contact_size,
        }
        for name, value in checks.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.rows < 2 or self.columns < 2:
            raise ValueError("rows and columns must each be >= 2.")

        grid_w = (self.columns - 1) * self.column_pitch
        grid_h = (self.rows - 1) * self.row_pitch
        if grid_w >= self.image_width or grid_h >= self.image_height:
            raise ValueError("Grid exceeds image dimensions.")

        if self.contact_size > min(self.row_pitch, self.column_pitch):
            raise ValueError("contact_size is too large for current pitch values.")

    def x_positions(self) -> list[int]:
        total_width = (self.columns - 1) * self.column_pitch
        start_x = (self.image_width - 1 - total_width) // 2
        return [start_x + i * self.column_pitch for i in range(self.columns)]

    def y_positions(self) -> list[int]:
        total_height = (self.rows - 1) * self.row_pitch
        start_y = (self.image_height - 1 - total_height) // 2
        return [start_y + i * self.row_pitch for i in range(self.rows)]
