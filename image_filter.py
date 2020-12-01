import numpy as np


def fill_temp_from_image(temp, image, row, column, color, row_offset, column_offset):
    rows, columns, _ = image.shape
    # If the filter is contained in the image, use the fast array-set
    if (
            row - row_offset >= 0
            and row + row_offset < rows
            and column - column_offset >= 0
            and column + column_offset < columns
    ):
        img_slice = image[
                         row - row_offset: row + row_offset + 1,
                         column - column_offset: column + column_offset + 1,
                         color
                    ]

        if img_slice.shape != temp.shape:
            img_slice.shape = temp.shape

        temp[:] = img_slice

    # Otherwise fill temp dynamically element by element
    else:
        row_min = row - min(row, row_offset)
        row_max = row + min(rows - row - 1, row_offset)
        column_min = column - min(column, column_offset)
        column_max = column + min(columns - column - 1, column_offset)

        for r in range(row_min, row_max):
            for c in range(column_min, column_max):
                temp[r - row + row_offset, c - column + column_offset] = image[r, c, color]


def apply_filter_to_row(row, image, flt):
    filter_2d = len(flt.shape) > 1
    rows, columns, colors = image.shape
    max_row_offset, max_column_offset = flt.shape[0] // 2, flt.shape[1] // 2 if filter_2d else 0

    temp = np.zeros(shape=flt.shape)
    temp_row = np.zeros(shape=(columns, colors))

    for column in range(columns):
        for color in range(colors):
            # reset temp
            temp.fill(0)

            fill_temp_from_image(temp, image, row, column, color, max_row_offset, max_column_offset)

            # apply filter to temp
            temp *= flt

            # calculate sum of filter for this pixel
            temp_row[column, color] = temp.sum()

    return temp_row
