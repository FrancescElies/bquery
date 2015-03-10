import numpy as np
import cython
from numpy cimport ndarray, dtype, npy_intp, npy_int32, npy_uint64, npy_int64, npy_float64
from libc.stdlib cimport malloc
from libc.string cimport strcpy
from khash cimport *
import bcolz as bz
from bcolz.carray_ext cimport carray, chunk

# ----------------------------------------------------------------------------
#                        GLOBAL DEFINITIONS
# ----------------------------------------------------------------------------

SUM = 0
DEF _SUM = 0

COUNT = 1
DEF _COUNT = 1

COUNT_NA = 2
DEF _COUNT_NA = 2

COUNT_DISTINCT = 3
DEF _COUNT_DISTINCT = 3

SORTED_COUNT_DISTINCT = 4
DEF _SORTED_COUNT_DISTINCT = 4
# ----------------------------------------------------------------------------

# Factorize Section
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_str_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       ndarray in_buffer,
                       ndarray[npy_uint64] out_buffer,
                       kh_str_t *table,
                       Py_ssize_t * count,
                       dict reverse,
                       ):
    cdef:
        Py_ssize_t i, idx
        int ret
        char * element
        char * insert
        khiter_t k

    ret = 0

    for i in range(iter_range):
        # TODO: Consider indexing directly into the array for efficiency
        element = in_buffer[i]
        k = kh_get_str(table, element)
        if k != table.n_buckets:
            idx = table.vals[k]
        else:
            # allocate enough memory to hold the string, add one for the
            # null byte that marks the end of the string.
            insert = <char *>malloc(allocation_size)
            # TODO: is strcpy really the best way to copy a string?
            strcpy(insert, element)
            k = kh_put_str(table, insert, &ret)
            table.vals[k] = idx = count[0]
            reverse[count[0]] = element
            count[0] += 1
        out_buffer[i] = idx

@cython.wraparound(False)
@cython.boundscheck(False)
def factorize_str(carray carray_, carray labels=None):
    cdef:
        Py_ssize_t len_carray, count, chunklen, len_in_buffer
        dict reverse
        ndarray in_buffer
        ndarray[npy_uint64] out_buffer
        kh_str_t *table

    count = 0
    ret = 0
    reverse = {}

    len_carray = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=len_carray)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    table = kh_init_str()

    for in_buffer in bz.iterblocks(carray_):
        len_in_buffer = len(in_buffer)
        _factorize_str_helper(len_in_buffer,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        table,
                        &count,
                        reverse,
                        )
        # compress out_buffer into labels
        labels.append(out_buffer[:len_in_buffer].astype(np.int64))

    kh_destroy_str(table)

    return labels, reverse

{% for factor_type in factor_types -%}
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _factorize_{{ factor_type }}_helper(Py_ssize_t iter_range,
                       Py_ssize_t allocation_size,
                       ndarray[npy_{{ factor_type }}] in_buffer,
                       ndarray[npy_uint64] out_buffer,
                       kh_{{ factor_type }}_t *table,
                       Py_ssize_t * count,
                       dict reverse,
                       ):
    cdef:
        Py_ssize_t i, idx
        int ret
        npy_{{ factor_type }} element
        khiter_t k

    ret = 0

    for i in range(iter_range):
        # TODO: Consider indexing directly into the array for efficiency
        element = in_buffer[i]
        k = kh_get_{{ factor_type }}(table, element)
        if k != table.n_buckets:
            idx = table.vals[k]
        else:
            k = kh_put_{{ factor_type }}(table, element, &ret)
            table.vals[k] = idx = count[0]
            reverse[count[0]] = element
            count[0] += 1
        out_buffer[i] = idx

@cython.wraparound(False)
@cython.boundscheck(False)
def factorize_{{ factor_type }}(carray carray_, carray labels=None):
    cdef:
        Py_ssize_t len_carray, count, chunklen, len_in_buffer
        dict reverse
        ndarray[npy_{{ factor_type }}] in_buffer
        ndarray[npy_uint64] out_buffer
        kh_{{ factor_type }}_t *table

    count = 0
    ret = 0
    reverse = {}

    len_carray = len(carray_)
    chunklen = carray_.chunklen
    if labels is None:
        labels = carray([], dtype='int64', expectedlen=len_carray)
    # in-buffer isn't typed, because cython doesn't support string arrays (?)
    out_buffer = np.empty(chunklen, dtype='uint64')
    table = kh_init_{{ factor_type }}()

    for in_buffer in bz.iterblocks(carray_):
        len_in_buffer = len(in_buffer)
        _factorize_{{ factor_type }}_helper(len_in_buffer,
                        carray_.dtype.itemsize + 1,
                        in_buffer,
                        out_buffer,
                        table,
                        &count,
                        reverse,
                        )
        # compress out_buffer into labels
        labels.append(out_buffer[:len_in_buffer].astype(np.int64))

    kh_destroy_{{ factor_type }}(table)

    return labels, reverse

{% endfor -%}

cpdef factorize(carray carray_, carray labels=None):
    if carray_.dtype == 'int32':
        labels, reverse = factorize_int32(carray_, labels=labels)
    elif carray_.dtype == 'int64':
        labels, reverse = factorize_int64(carray_, labels=labels)
    elif carray_.dtype == 'float64':
        labels, reverse = factorize_float64(carray_, labels=labels)
    else:
        #TODO: check that the input is a string_ dtype type
        labels, reverse = factorize_str(carray_, labels=labels)
    return labels, reverse

# ---------------------------------------------------------------------------
# Translate existing arrays
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef translate_int64(carray input_, carray output_, dict lookup, npy_int64 default=-1):
    cdef:
        Py_ssize_t chunklen, leftover_elements, len_in_buffer
        ndarray[npy_int64] in_buffer
        ndarray[npy_int64] out_buffer

    chunklen = input_.chunklen
    out_buffer = np.empty(chunklen, dtype='int64')

    for in_buffer in bz.iterblocks(input_):
        len_in_buffer = len(in_buffer)
        for i in range(len_in_buffer):
            element = in_buffer[i]
            out_buffer[i] = lookup.get(element, default)
        # compress out_buffer into labels
        output_.append(out_buffer[:len_in_buffer].astype(np.int64))

# ---------------------------------------------------------------------------
# Aggregation Section (old)
@cython.boundscheck(False)
@cython.wraparound(False)
def agg_sum_na(iter_):
    cdef:
        npy_float64 v, v_cum = 0.0

    for v in iter_:
        if v == v:  # skip NA values
            v_cum += v

    return v_cum

@cython.boundscheck(False)
@cython.wraparound(False)
def agg_sum(iter_):
    cdef:
        npy_float64 v, v_cum = 0.0

    for v in iter_:
        v_cum += v

    return v_cum

# ---------------------------------------------------------------------------
# Aggregation Section
@cython.boundscheck(False)
@cython.wraparound(False)
def groupsort_indexer(carray index, Py_ssize_t ngroups):
    cdef:
        Py_ssize_t i, label, n, len_in_buffer
        ndarray[int64_t] counts, where, np_result
        # --
        carray c_result
        chunk input_chunk, index_chunk
        Py_ssize_t index_chunk_nr, index_chunk_len, leftover_elements

        ndarray[int64_t] in_buffer

    index_chunk_len = index.chunklen
    in_buffer = np.empty(index_chunk_len, dtype='int64')
    index_chunk_nr = 0

    # count group sizes, location 0 for NA
    counts = np.zeros(ngroups + 1, dtype=np.int64)
    n = len(index)

    for in_buffer in bz.iterblocks(index):
        len_in_buffer = len(in_buffer)
        # loop through rows
        for i in range(len_in_buffer):
            counts[index[i] + 1] += 1

    # mark the start of each contiguous group of like-indexed data
    where = np.zeros(ngroups + 1, dtype=np.int64)
    for i from 1 <= i < ngroups + 1:
        where[i] = where[i - 1] + counts[i - 1]

    # this is our indexer
    np_result = np.zeros(n, dtype=np.int64)
    for i from 0 <= i < n:
        label = index[i] + 1
        np_result[where[label]] = i
        where[label] += 1

    return np_result, counts

{% for count_unique_type in count_unique_types -%}
cdef count_unique_{{ count_unique_type }}(ndarray[{{ count_unique_type }}_t] values):
    cdef:
        Py_ssize_t i, n = len(values)
        Py_ssize_t idx
        int ret = 0
        {{ count_unique_type }}_t val
        khiter_t k
        npy_uint64 count = 0
{%- if count_unique_type == "float64" %}
        bint seen_na = 0
{%- endif %}
        kh_{{ count_unique_type }}_t *table

    table = kh_init_{{ count_unique_type }}()

    for i in range(n):
        val = values[i]

        if val == val:
            k = kh_get_{{ count_unique_type }}(table, val)
            if k == table.n_buckets:
                k = kh_put_{{ count_unique_type }}(table, val, &ret)
                count += 1
{%- if count_unique_type == "float64" %}
        elif not seen_na:
            seen_na = 1
            count += 1
{%- endif %}

    kh_destroy_{{ count_unique_type }}(table)

    return count

{% endfor -%}

{% for sum_type in sum_types -%}
@cython.wraparound(False)
@cython.boundscheck(False)
cdef sum_{{ sum_type }}(carray ca_input, carray ca_factor,
               Py_ssize_t nr_groups, Py_ssize_t skip_key, agg_method=_SUM):
    cdef:
        Py_ssize_t in_buffer_len, factor_buffer_len
        Py_ssize_t factor_chunk_nr, factor_chunk_row
        Py_ssize_t current_index, i, j, end_counts, start_counts

        ndarray[npy_{{ sum_type }}] in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray[npy_{{ sum_type }}] out_buffer
        ndarray[npy_{{ sum_type }}] last_values

        npy_{{ sum_type }} v
        bint count_distinct_started = 0
        carray num_uniques

    count = 0
    ret = 0
    reverse = {}
    iter_ca_factor = bz.iterblocks(ca_factor)

    if agg_method == _COUNT_DISTINCT:
        num_uniques = carray([], dtype='int64')
        positions, counts = groupsort_indexer(ca_factor, nr_groups)
        start_counts = 0
        end_counts = 0
        for j in range(len(counts) - 1):
            start_counts = end_counts
            end_counts = start_counts + counts[j + 1]
            positions[start_counts:end_counts]
            num_uniques.append(
                count_unique_{{ sum_type }}(ca_input[positions[start_counts:end_counts]])
            )

        return num_uniques

    factor_chunk_nr = 0
    factor_buffer = iter_ca_factor.next()
    factor_buffer_len = len(factor_buffer)
    factor_chunk_row = 0
    out_buffer = np.zeros(nr_groups, dtype='{{ sum_type }}')

    for in_buffer in bz.iterblocks(ca_input):
        in_buffer_len = len(in_buffer)

        # loop through rows
        for i in range(in_buffer_len):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_buffer_len:
                factor_chunk_nr += 1
                factor_buffer = iter_ca_factor.next()
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                if agg_method == _SUM:
                    out_buffer[current_index] += in_buffer[i]
                elif agg_method == _COUNT:
                    out_buffer[current_index] += 1
                elif agg_method == _COUNT_NA:
{% if sum_type == "float64" %}
                    v = in_buffer[i]
                    if v == v:  # skip NA values
                        out_buffer[current_index] += 1
{%- else %}
                    # TODO: Warning: int does not support NA values, is this what we need?
                    out_buffer[current_index] += 1
{%- endif %}
                elif agg_method == _SORTED_COUNT_DISTINCT:
                    v = in_buffer[i]
                    if not count_distinct_started:
                        count_distinct_started = 1
                        last_values = np.zeros(nr_groups, dtype='{{ sum_type }}')
                        last_values[0] = v
                        out_buffer[0] = 1
                    else:
                        if v != last_values[current_index]:
                            out_buffer[current_index] += 1

                    last_values[current_index] = v
                else:
                    raise NotImplementedError('sumtype not supported')

    # check whether a row has to be removed if it was meant to be skipped
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

    return out_buffer

{% endfor -%}

@cython.wraparound(False)
@cython.boundscheck(False)
cdef groupby_value(carray ca_input, carray ca_factor, Py_ssize_t nr_groups, Py_ssize_t skip_key):
    cdef:
        Py_ssize_t in_buffer_len, factor_buffer_len
        Py_ssize_t factor_chunk_nr, factor_chunk_row
        Py_ssize_t current_index, i, factor_total_chunks

        ndarray in_buffer
        ndarray[npy_int64] factor_buffer
        ndarray out_buffer

    count = 0
    ret = 0
    reverse = {}
    iter_ca_factor = bz.iterblocks(ca_factor)


    factor_total_chunks = ca_factor.nchunks
    factor_chunk_nr = 0
    factor_buffer = iter_ca_factor.next()
    factor_buffer_len = len(factor_buffer)
    factor_chunk_row = 0
    out_buffer = np.zeros(nr_groups, dtype=ca_input.dtype)

    for in_buffer in bz.iterblocks(ca_input):
        in_buffer_len = len(in_buffer)

        for i in range(in_buffer_len):

            # go to next factor buffer if necessary
            if factor_chunk_row == factor_buffer_len:
                factor_chunk_nr += 1
                factor_buffer = iter_ca_factor.next()
                factor_chunk_row = 0

            # retrieve index
            current_index = factor_buffer[factor_chunk_row]
            factor_chunk_row += 1

            # update value if it's not an invalid index
            if current_index != skip_key:
                out_buffer[current_index] = in_buffer[i]

    # check whether a row has to be fixed
    if skip_key < nr_groups:
        np.delete(out_buffer, skip_key)

    return out_buffer

def aggregate_groups_by_iter_2(ct_input,
                        ct_agg,
                        npy_uint64 nr_groups,
                        npy_uint64 skip_key,
                        carray factor_carray,
                        groupby_cols,
                        output_agg_ops,
                        dtype_list,
                        agg_method=_SUM
                        ):
    total = []

    for col in groupby_cols:
        total.append(groupby_value(ct_input[col], factor_carray, nr_groups, skip_key))

    for col, agg_op in output_agg_ops:
        # TODO: input vs output column
        col_dtype = ct_agg[col].dtype
        if col_dtype == np.float64:
            total.append(
                sum_float64(ct_input[col], factor_carray, nr_groups, skip_key,
                            agg_method=agg_method)
            )
        elif col_dtype == np.int64:
            total.append(
                sum_int64(ct_input[col], factor_carray, nr_groups, skip_key,
                          agg_method=agg_method)
            )
        elif col_dtype == np.int32:
            total.append(
                sum_int32(ct_input[col], factor_carray, nr_groups, skip_key,
                          agg_method=agg_method)
            )
        else:
            raise NotImplementedError(
                'Column dtype ({0}) not supported for aggregation yet '
                '(only int32, int64 & float64)'.format(str(col_dtype)))

    ct_agg.append(total)

# ---------------------------------------------------------------------------
# Temporary Section
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef carray_is_in(carray col, set value_set, ndarray boolarr, bint reverse):
    """
    TEMPORARY WORKAROUND till numexpr support in list operations

    Update a boolean array with checks whether the values of a column (col) are in a set (value_set)
    Reverse means "not in" functionality

    For the 0d array work around, see https://github.com/Blosc/bcolz/issues/61

    :param col:
    :param value_set:
    :param boolarr:
    :param reverse:
    :return:
    """
    cdef Py_ssize_t i
    i = 0
    if not reverse:
        for val in col.iter():
            if val not in value_set:
                boolarr[i] = False
            i += 1
    else:
        for val in col.iter():
            if val in value_set:
                boolarr[i] = False
            i += 1

