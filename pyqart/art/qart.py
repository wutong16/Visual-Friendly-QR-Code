# Added at : 2016.8.2
# Author   : 7sDream
# Usage    : Accept data and source image, make a QArt.

import itertools
import cv2
from random import randint

from .source import QArtSourceImage
from .bitblock import BitBlock
from ..qr import QrData, QrPainter
from ..qr.data.numbers import Numbers
from ..qr.painter.point import QrPointType
from ..qr.ec import RSEncoder
from ..common import Bits, BIT_PER_CW




__all__ = ['QArtist']

INF = float('inf')


class QArtist(QrPainter):
    def __init__(self, url, img, version=6, mask=None, level=0, rotation=0,
                 dither=False, only_data=False, rand=False, higher_first=False,
                 dy=None, dx=None, prefer_option = False, prefer_mask = None):
        assert isinstance(img, (str, QArtSourceImage))
        if isinstance(img, str):
            img = QArtSourceImage(img)
        self.source = img
        self.dy = dy
        self.dx = dx
        self._only_data = bool(only_data)
        self._higher_first = bool(higher_first)
        self.prefer_option = prefer_option
        if prefer_option:
            self.prefer_mask = cv2.resize(cv2.imread(prefer_mask),(9*(version*4 + 17),9*(version*4 + 17)))
            self.prefer_mask = cv2.cvtColor(self.prefer_mask, cv2.COLOR_BGR2GRAY)
            self.prefer_mask = cv2.blur(self.prefer_mask, (200, 200))
        data = QrData(url + '#', level)
        super().__init__(data, version, mask, rotation)
        args, _, _ = self.get_params()
        # print('# Processing input image...', end='', flush=True)
        self._targets = self.source.to_targets(
            self.canvas, args, bool(dither), rand, dy, dx)

        # test_targets = self.source.to_targets(
        #     self.canvas, args, bool(dither), rand, dy, dx)
        self.dither = dither
        # print('Done')
        self._bits = None
        self.failure_time = 0
        self.failure = False

    @property
    def bits(self):
        # print('# Create BitBlock')
        if self._bits is not None:
            return self._bits
        args, available, used = self.get_params()
        cci_length = args.cci_length_of(Numbers)
        available_for_number = available - 4 - cci_length
        used += 4 + cci_length
        if available_for_number < 4:
            return super().bits
        else:
            numbers_count = available_for_number // 10 * 3
            remaining = available_for_number % 10
            if remaining >= 7:
                numbers_count += 2
                remaining -= 7
            elif remaining >= 4:
                numbers_count += 1
                remaining -= 4
            upper = args.dcwc * BIT_PER_CW - remaining

        self._data.put_numbers('0' * numbers_count)

        while True:
            bits = super().bits
            di = 0
            eci = args.dcwc * BIT_PER_CW
            ecbc = args.eccwcpb * BIT_PER_CW
            data_bits = Bits()
            ec_bits = Bits()

            for i in range(args.bc):
                dbc = args.dcwcof(i) * BIT_PER_CW
                low = 0
                high = dbc
                if di < used:
                    low = used - di
                    if low >= dbc:
                        data_bits.extend(bits, di, dbc)
                        ec_bits.extend(bits, eci, ecbc)
                        di += dbc
                        eci += ecbc
                        continue

                if di + dbc > upper:
                    high = upper - di
                    if high <= 0:
                        data_bits.extend(bits, di, dbc)
                        ec_bits.extend(bits, eci, ecbc)
                        di += dbc
                        eci += ecbc
                        continue

                if not self._only_data:
                    # print('Create BitBlock', '{i}/{bc}...'.format(
                    #     i=i+1, bc=args.bc,
                    # ), end='', flush=True)
                    block = BitBlock(bits, di, dbc, eci, ecbc)
                else:
                    block = Bits.copy_from(bits, di, dbc)

                # Lock uncontrollable bits

                locked_bits = set()

                if not self._only_data:
                    for j in itertools.chain(range(0, low), range(high, dbc)):
                        assert block.set(j, bits[di + j])
                else:
                    for j in itertools.chain(range(0, low), range(high, dbc)):
                        locked_bits.add(j)

                targets_index = list(range(di, di+dbc))
                if not self._only_data:
                    targets_index.extend(range(eci, eci+ecbc))

                # ******************************key********************************
                def compare_l2_location(x):
                    t = self._targets[x]
                    center = (self._version*4 + 17)/2
                    distance = (t.x - center)**2 + (t.y - center)**2
                    return distance

                def compare_l1_location(x):
                    t = self._targets[x]
                    center = (self._version*4 + 17)/2
                    distance = max(abs(t.x - center) , abs(t.y - center))
                    return distance

                def compare_prefer(x):
                    # prefer_mask is a grayscale image where the darker it is, the more the region is prefered
                    t = self._targets[x]
                    s = -(255 - self.prefer_mask[t.x,t.y])+0.0001*compare_l2_location(x)
                    return s


                def compare(x):
                    t = self._targets[x]
                    if t.is_hard_zero():
                        if self._higher_first:
                            return INF
                        else:
                            return -1
                    else:
                        return t.contrast

                if not self.prefer_option:
                    targets_index = sorted(targets_index, key=compare_l1_location,
                                       reverse=self._higher_first)
                else:
                    targets_index = sorted(targets_index, key=compare_prefer,
                                           reverse=self._higher_first)


                for target_index in targets_index:
                    target = self._targets[target_index]
                    point = target.point
                    fill = target.fill
                    if point.invert:
                        fill = not fill
                    if target.is_hard_zero():
                        fill = False
                    if point.type is QrPointType.DATA:
                        index = point.offset - di
                    else:
                        assert point.type is QrPointType.CORRECTION
                        index = point.offset - eci + dbc
                    if not self._only_data:
                        block.set(index, fill)
                    elif index not in locked_bits:
                        block[index] = fill

                if not self._only_data:
                    new_block_bits = block.bits()
                    data_bits.extend(new_block_bits, 0, dbc)
                    ec_bits.extend(new_block_bits, dbc, ecbc)
                else:
                    data_bits.extend(block)
                    ec_bits.extend(RSEncoder.encode(block, ecbc // 8, True))

                di += dbc
                eci += ecbc

            error_count = 0

            numbers = ''
            for i in range(0, numbers_count, 3):
                if i + 3 > numbers_count:
                    count = [None, 4, 7][numbers_count - i]
                else:
                    count = 10
                offset = used + i // 3 * 10
                value = Bits.copy_from(data_bits, offset, count)
                value = value.as_int
                if count == 10 and value >= 1000:
                    rand_pos = randint(0, 4)
                    hard_zero_pos = offset + rand_pos
                    self._targets[hard_zero_pos].set_hard_zero()
                    error_count += 1
                    value -= 2**(9 - rand_pos)
                elif count == 7 and value >= 100:
                    rand_pos = randint(0, 1)
                    hard_zero_pos = offset + rand_pos
                    self._targets[hard_zero_pos].set_hard_zero()
                    error_count += 1
                    value -= 2**(6 - rand_pos)
                elif count == 4 and value >= 10:
                    hard_zero_pos = offset
                    self._targets[hard_zero_pos].set_hard_zero()
                    error_count += 1
                    value -= 8
                numbers += str(value).rjust(count // 3, '0')

            # print('Error count', error_count, end='')
            if error_count == 0:
                # print(', send to printer.')
                data_bits.extend(ec_bits)
                self._bits = data_bits
                return data_bits
            else:
                self.failure_time += 1
                if self.failure_time == 7:
                    # print(', rebuid the bits')
                    break
                # print(', restart.')
        # 如果失败次数过多，说明要重新构造bits
        if self.failure_time == 7:
            self.failure = True
            return self._bits


