"""Dense primal-dual SOCP solver and C ABI.

The solver uses the Euclidean Jordan algebra of the nonnegative and Lorentz
cones. All storage is owned by the caller; matrices are row-major float64.
"""

from std.algorithm.functional import parallelize
from std.math import abs, sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simdwidthof[DType.float64]()
comptime PARALLEL_MATVEC_WORK = 131072
comptime PARALLEL_ELIMINATION_WORK = 1048576
comptime PARALLEL_WORKERS = 16


def norm_inf(x: Ptr, n: Int) -> Float64:
    var values = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        values = max(values, abs(x.load[width=W](i)))
        i += W
    var value = values.reduce_max()
    while i < n:
        value = max(value, abs(x[i]))
        i += 1
    return value


def dot(x: Ptr, y: Ptr, n: Int) -> Float64:
    var values = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        values += x.load[width=W](i) * y.load[width=W](i)
        i += W
    var value = values.reduce_add()
    while i < n:
        value += x[i] * y[i]
        i += 1
    return value


def cone_dot(x: Ptr, y: Ptr, m: Int) -> Float64:
    return dot(x, y, m)


def cone_identity(x: Ptr, l: Int, q: IPtr, nq: Int):
    var ones = SIMD[DType.float64, W](1.0)
    var i = 0
    while i + W <= l:
        x.store(i, ones)
        i += W
    while i < l:
        x[i] = 1.0
        i += 1
    var offset = l
    for cone in range(nq):
        var size = Int(q[cone])
        x[offset] = 1.0
        var zeros = SIMD[DType.float64, W](0.0)
        var j = 1
        while j + W <= size:
            x.store(offset + j, zeros)
            j += W
        while j < size:
            x[offset + j] = 0.0
            j += 1
        offset += size


def cone_product(a: Ptr, b: Ptr, dst: Ptr, l: Int, q: IPtr, nq: Int):
    var i = 0
    while i + W <= l:
        dst.store(
            i,
            a.load[width=W](i) * b.load[width=W](i),
        )
        i += W
    while i < l:
        dst[i] = a[i] * b[i]
        i += 1
    var offset = l
    for cone in range(nq):
        var size = Int(q[cone])
        var products = SIMD[DType.float64, W](0.0)
        var j = 1
        while j + W <= size:
            products += (
                a.load[width=W](offset + j)
                * b.load[width=W](offset + j)
            )
            j += W
        var scalar = a[offset] * b[offset] + products.reduce_add()
        while j < size:
            scalar += a[offset + j] * b[offset + j]
            j += 1
        dst[offset] = scalar
        var a0 = SIMD[DType.float64, W](a[offset])
        var b0 = SIMD[DType.float64, W](b[offset])
        j = 1
        while j + W <= size:
            dst.store(
                offset + j,
                a0 * b.load[width=W](offset + j)
                + b0 * a.load[width=W](offset + j),
            )
            j += W
        while j < size:
            dst[offset + j] = (
                a[offset] * b[offset + j] + b[offset] * a[offset + j]
            )
            j += 1
        offset += size


def cone_interior(x: Ptr, l: Int, q: IPtr, nq: Int) -> Bool:
    for i in range(l):
        if x[i] <= 0.0:
            return False
    var offset = l
    for cone in range(nq):
        var size = Int(q[cone])
        var squared = 0.0
        for j in range(1, size):
            squared += x[offset + j] * x[offset + j]
        if x[offset] <= 0.0 or x[offset] * x[offset] <= squared:
            return False
        offset += size
    return True


def cone_step(x: Ptr, dx: Ptr, l: Int, q: IPtr, nq: Int) -> Float64:
    var upper = 1.0
    for i in range(l):
        if dx[i] < 0.0:
            upper = min(upper, -x[i] / dx[i])
    if nq == 0:
        return upper

    var feasible = True
    var offset = l
    for cone in range(nq):
        var size = Int(q[cone])
        var squares = SIMD[DType.float64, W](0.0)
        var j = 1
        while j + W <= size:
            var values = (
                x.load[width=W](offset + j)
                + SIMD[DType.float64, W](upper)
                * dx.load[width=W](offset + j)
            )
            squares += values * values
            j += W
        var squared = squares.reduce_add()
        while j < size:
            var value = x[offset + j] + upper * dx[offset + j]
            squared += value * value
            j += 1
        var scalar = x[offset] + upper * dx[offset]
        if scalar <= 0.0 or scalar * scalar <= squared:
            feasible = False
        offset += size
    if feasible:
        return upper

    var lo = 0.0
    var hi = upper
    for _ in range(55):
        var alpha = 0.5 * (lo + hi)
        feasible = True
        for i in range(l):
            if x[i] + alpha * dx[i] <= 0.0:
                feasible = False
        offset = l
        for cone in range(nq):
            var size = Int(q[cone])
            var squares = SIMD[DType.float64, W](0.0)
            var j = 1
            while j + W <= size:
                var values = (
                    x.load[width=W](offset + j)
                    + SIMD[DType.float64, W](alpha)
                    * dx.load[width=W](offset + j)
                )
                squares += values * values
                j += W
            var squared = squares.reduce_add()
            while j < size:
                var value = x[offset + j] + alpha * dx[offset + j]
                squared += value * value
                j += 1
            var scalar = x[offset] + alpha * dx[offset]
            if scalar <= 0.0 or scalar * scalar <= squared:
                feasible = False
            offset += size
        if feasible:
            lo = alpha
        else:
            hi = alpha
    return lo


def residuals(
    c: Ptr,
    g: Ptr,
    h: Ptr,
    a: Ptr,
    b: Ptr,
    x: Ptr,
    y: Ptr,
    z: Ptr,
    s: Ptr,
    rd: Ptr,
    rp: Ptr,
    rg: Ptr,
    n: Int,
    m: Int,
    p: Int,
):
    var col = 0
    while col + W <= n:
        rd.store(col, c.load[width=W](col))
        col += W
    while col < n:
        rd[col] = c[col]
        col += 1
    for row in range(m):
        var scale = SIMD[DType.float64, W](z[row])
        col = 0
        while col + W <= n:
            rd.store(
                col,
                rd.load[width=W](col)
                + scale * g.load[width=W](row * n + col),
            )
            col += W
        while col < n:
            rd[col] += z[row] * g[row * n + col]
            col += 1
    for row in range(p):
        var scale = SIMD[DType.float64, W](y[row])
        col = 0
        while col + W <= n:
            rd.store(
                col,
                rd.load[width=W](col)
                + scale * a.load[width=W](row * n + col),
            )
            col += W
        while col < n:
            rd[col] += y[row] * a[row * n + col]
            col += 1

    @parameter
    def compute_rp(row: Int):
        rp[row] = -b[row] + dot(a + row * n, x, n)

    @parameter
    def compute_rg(row: Int):
        rg[row] = s[row] - h[row] + dot(g + row * n, x, n)

    if p * n >= PARALLEL_MATVEC_WORK:
        parallelize[compute_rp](p, PARALLEL_WORKERS)
    else:
        for row in range(p):
            compute_rp(row)
    if m * n >= PARALLEL_MATVEC_WORK:
        parallelize[compute_rg](m, PARALLEL_WORKERS)
    else:
        for row in range(m):
            compute_rg(row)


def fill_newton(
    g: Ptr,
    a: Ptr,
    s: Ptr,
    z: Ptr,
    rd: Ptr,
    rp: Ptr,
    rg: Ptr,
    correction: Ptr,
    matrix: Ptr,
    rhs: Ptr,
    q: IPtr,
    n: Int,
    m: Int,
    p: Int,
    l: Int,
    nq: Int,
    target_mu: Float64,
    use_correction: Bool,
    build_matrix: Bool,
):
    var dim = n + p + m
    if build_matrix:
        var zeros = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= dim * dim:
            matrix.store(i, zeros)
            i += W
        while i < dim * dim:
            matrix[i] = 0.0
            i += 1

    for col in range(n):
        rhs[col] = -rd[col]
        if build_matrix:
            for row in range(p):
                matrix[col * dim + n + row] = a[row * n + col]
            for row in range(m):
                matrix[col * dim + n + p + row] = g[row * n + col]

    for row in range(p):
        var rr = n + row
        rhs[rr] = -rp[row]
        if build_matrix:
            var col = 0
            while col + W <= n:
                matrix.store(
                    rr * dim + col,
                    a.load[width=W](row * n + col),
                )
                col += W
            while col < n:
                matrix[rr * dim + col] = a[row * n + col]
                col += 1

    for i in range(l):
        var rr = n + p + i
        if build_matrix:
            var scale = SIMD[DType.float64, W](-z[i])
            var col = 0
            while col + W <= n:
                matrix.store(
                    rr * dim + col,
                    scale * g.load[width=W](i * n + col),
                )
                col += W
            while col < n:
                matrix[rr * dim + col] = -z[i] * g[i * n + col]
                col += 1
            matrix[rr * dim + n + p + i] = s[i]
        var corr = correction[i] if use_correction else 0.0
        rhs[rr] = -s[i] * z[i] + target_mu - corr + z[i] * rg[i]

    var offset = l
    for cone in range(nq):
        var size = Int(q[cone])
        for local_row in range(size):
            var rr = n + p + offset + local_row
            if build_matrix:
                var col = 0
                if local_row == 0:
                    while col + W <= n:
                        var lzg = (
                            SIMD[DType.float64, W](z[offset])
                            * g.load[width=W](offset * n + col)
                        )
                        for j in range(1, size):
                            lzg += (
                                SIMD[DType.float64, W](z[offset + j])
                                * g.load[width=W]((offset + j) * n + col)
                            )
                        matrix.store(rr * dim + col, -lzg)
                        col += W
                    while col < n:
                        var lzg = z[offset] * g[offset * n + col]
                        for j in range(1, size):
                            lzg += z[offset + j] * g[(offset + j) * n + col]
                        matrix[rr * dim + col] = -lzg
                        col += 1
                else:
                    var zlocal = SIMD[DType.float64, W](
                        z[offset + local_row]
                    )
                    var zscalar = SIMD[DType.float64, W](z[offset])
                    while col + W <= n:
                        matrix.store(
                            rr * dim + col,
                            -(
                                zlocal * g.load[width=W](offset * n + col)
                                + zscalar
                                * g.load[width=W](
                                    (offset + local_row) * n + col
                                )
                            ),
                        )
                        col += W
                    while col < n:
                        matrix[rr * dim + col] = -(
                            z[offset + local_row] * g[offset * n + col]
                            + z[offset] * g[(offset + local_row) * n + col]
                        )
                        col += 1

                if local_row == 0:
                    matrix[rr * dim + n + p + offset] = s[offset]
                    for j in range(1, size):
                        matrix[rr * dim + n + p + offset + j] = s[offset + j]
                else:
                    matrix[rr * dim + n + p + offset] = s[
                        offset + local_row
                    ]
                    matrix[
                        rr * dim + n + p + offset + local_row
                    ] = s[offset]

            var sz = 0.0
            var lzrg = 0.0
            var corr = correction[offset + local_row] if use_correction else 0.0
            if local_row == 0:
                sz = s[offset] * z[offset]
                lzrg = z[offset] * rg[offset]
                for j in range(1, size):
                    sz += s[offset + j] * z[offset + j]
                    lzrg += z[offset + j] * rg[offset + j]
                rhs[rr] = -sz + 2.0 * target_mu - corr + lzrg
            else:
                sz = (
                    s[offset] * z[offset + local_row]
                    + z[offset] * s[offset + local_row]
                )
                lzrg = (
                    z[offset + local_row] * rg[offset]
                    + z[offset] * rg[offset + local_row]
                )
                rhs[rr] = -sz - corr + lzrg
        offset += size


def fill_newton_linear(
    g: Ptr,
    a: Ptr,
    s: Ptr,
    z: Ptr,
    rd: Ptr,
    rp: Ptr,
    rg: Ptr,
    correction: Ptr,
    matrix: Ptr,
    rhs: Ptr,
    n: Int,
    m: Int,
    p: Int,
    target_mu: Float64,
    use_correction: Bool,
    build_matrix: Bool,
):
    var dim = n + p
    if build_matrix:
        var zeros = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= dim * dim:
            matrix.store(i, zeros)
            i += W
        while i < dim * dim:
            matrix[i] = 0.0
            i += 1

    var col = 0
    while col + W <= n:
        rhs.store(col, -rd.load[width=W](col))
        col += W
    while col < n:
        rhs[col] = -rd[col]
        col += 1
    for row in range(p):
        rhs[n + row] = -rp[row]
        if build_matrix:
            for col in range(n):
                matrix[col * dim + n + row] = a[row * n + col]
                matrix[(n + row) * dim + col] = a[row * n + col]

    for row in range(m):
        var corr = correction[row] if use_correction else 0.0
        var cone_rhs = (
            -s[row] * z[row]
            + target_mu
            - corr
            + z[row] * rg[row]
        )
        rhs[dim + row] = cone_rhs
        var inverse_s = 1.0 / s[row]
        var rhs_scale = SIMD[DType.float64, W](cone_rhs * inverse_s)
        col = 0
        while col + W <= n:
            rhs.store(
                col,
                rhs.load[width=W](col)
                - rhs_scale * g.load[width=W](row * n + col),
            )
            col += W
        while col < n:
            rhs[col] -= cone_rhs * inverse_s * g[row * n + col]
            col += 1

        if build_matrix:
            var weight = z[row] * inverse_s
            for left in range(n):
                var factor = weight * g[row * n + left]
                if factor != 0.0:
                    var factors = SIMD[DType.float64, W](factor)
                    col = 0
                    while col + W <= n:
                        matrix.store(
                            left * dim + col,
                            matrix.load[width=W](left * dim + col)
                            + factors
                            * g.load[width=W](row * n + col),
                        )
                        col += W
                    while col < n:
                        matrix[left * dim + col] += (
                            factor * g[row * n + col]
                        )
                        col += 1


def recover_dz_linear(
    g: Ptr,
    s: Ptr,
    z: Ptr,
    cone_rhs: Ptr,
    dx: Ptr,
    dz: Ptr,
    n: Int,
    m: Int,
):
    @parameter
    def compute_row(row: Int):
        dz[row] = (
            cone_rhs[row] + z[row] * dot(g + row * n, dx, n)
        ) / s[row]

    if m * n >= PARALLEL_MATVEC_WORK:
        parallelize[compute_row](m, PARALLEL_WORKERS)
    else:
        for row in range(m):
            compute_row(row)


def lorentz_solve(s: Ptr, value: Ptr, result: Ptr, size: Int):
    var scalar = s[0]
    var determinant = scalar * scalar - dot(s + 1, s + 1, size - 1)
    var result_scalar = (
        scalar * value[0] - dot(s + 1, value + 1, size - 1)
    ) / determinant
    result[0] = result_scalar
    var scalars = SIMD[DType.float64, W](scalar)
    var result_scalars = SIMD[DType.float64, W](result_scalar)
    var j = 1
    while j + W <= size:
        result.store(
            j,
            (
                value.load[width=W](j)
                - s.load[width=W](j) * result_scalars
            ) / scalars,
        )
        j += W
    while j < size:
        result[j] = (value[j] - s[j] * result_scalar) / scalar
        j += 1


def fill_newton_reduced_soc(
    g: Ptr,
    a: Ptr,
    s: Ptr,
    z: Ptr,
    rd: Ptr,
    rp: Ptr,
    rg: Ptr,
    correction: Ptr,
    matrix: Ptr,
    rhs: Ptr,
    q: IPtr,
    n: Int,
    m: Int,
    p: Int,
    l: Int,
    nq: Int,
    target_mu: Float64,
    use_correction: Bool,
    build_matrix: Bool,
):
    var dim = n + p
    var work = matrix + dim * dim
    var product = work + m * W
    if build_matrix:
        var zeros = SIMD[DType.float64, W](0.0)
        var i = 0
        while i + W <= dim * dim:
            matrix.store(i, zeros)
            i += W
        while i < dim * dim:
            matrix[i] = 0.0
            i += 1

    var col = 0
    while col + W <= n:
        rhs.store(col, -rd.load[width=W](col))
        col += W
    while col < n:
        rhs[col] = -rd[col]
        col += 1
    for row in range(p):
        rhs[n + row] = -rp[row]
        if build_matrix:
            for col in range(n):
                matrix[col * dim + n + row] = a[row * n + col]
                matrix[(n + row) * dim + col] = a[row * n + col]

    for row in range(l):
        var corr = correction[row] if use_correction else 0.0
        var cone_rhs = (
            -s[row] * z[row]
            + target_mu
            - corr
            + z[row] * rg[row]
        )
        rhs[dim + row] = cone_rhs
        var inverse_s = 1.0 / s[row]
        var rhs_scale = SIMD[DType.float64, W](cone_rhs * inverse_s)
        col = 0
        while col + W <= n:
            rhs.store(
                col,
                rhs.load[width=W](col)
                - rhs_scale * g.load[width=W](row * n + col),
            )
            col += W
        while col < n:
            rhs[col] -= cone_rhs * inverse_s * g[row * n + col]
            col += 1
        if build_matrix:
            var weight = z[row] * inverse_s
            for left in range(n):
                var factor = weight * g[row * n + left]
                if factor != 0.0:
                    var factors = SIMD[DType.float64, W](factor)
                    col = 0
                    while col + W <= n:
                        matrix.store(
                            left * dim + col,
                            matrix.load[width=W](left * dim + col)
                            + factors
                            * g.load[width=W](row * n + col),
                        )
                        col += W
                    while col < n:
                        matrix[left * dim + col] += (
                            factor * g[row * n + col]
                        )
                        col += 1

    var offset = l
    for cone in range(nq):
        var size = Int(q[cone])
        var raw = rhs + dim + offset
        var corr = correction[offset] if use_correction else 0.0
        var sz = s[offset] * z[offset]
        var lzrg = z[offset] * rg[offset]
        for j in range(1, size):
            sz += s[offset + j] * z[offset + j]
            lzrg += z[offset + j] * rg[offset + j]
        raw[0] = -sz + 2.0 * target_mu - corr + lzrg
        for j in range(1, size):
            corr = correction[offset + j] if use_correction else 0.0
            raw[j] = -(
                s[offset] * z[offset + j]
                + z[offset] * s[offset + j]
            ) - corr + (
                z[offset + j] * rg[offset]
                + z[offset] * rg[offset + j]
            )

        lorentz_solve(s + offset, raw, work, size)
        for j in range(size):
            var scale = SIMD[DType.float64, W](work[j])
            col = 0
            while col + W <= n:
                rhs.store(
                    col,
                    rhs.load[width=W](col)
                    - scale * g.load[width=W]((offset + j) * n + col),
                )
                col += W
            while col < n:
                rhs[col] -= work[j] * g[(offset + j) * n + col]
                col += 1

        if build_matrix:
            var determinant = (
                s[offset] * s[offset]
                - dot(s + offset + 1, s + offset + 1, size - 1)
            )
            var s0 = SIMD[DType.float64, W](s[offset])
            var z0 = SIMD[DType.float64, W](z[offset])
            var determinants = SIMD[DType.float64, W](determinant)
            var right = 0
            while right + W <= n:
                var g0 = g.load[width=W](offset * n + right)
                var product_scalar = z0 * g0
                for j in range(1, size):
                    var gj = g.load[width=W]((offset + j) * n + right)
                    product_scalar += (
                        SIMD[DType.float64, W](z[offset + j]) * gj
                    )
                    product.store(
                        j * W,
                        SIMD[DType.float64, W](z[offset + j]) * g0
                        + z0 * gj,
                    )
                product.store(0, product_scalar)

                var s_dot_product = SIMD[DType.float64, W](0.0)
                for j in range(1, size):
                    s_dot_product += (
                        SIMD[DType.float64, W](s[offset + j])
                        * product.load[width=W](j * W)
                    )
                var result_scalar = (
                    s0 * product_scalar - s_dot_product
                ) / determinants
                work.store(0, result_scalar)
                for j in range(1, size):
                    work.store(
                        j * W,
                        (
                            product.load[width=W](j * W)
                            - SIMD[DType.float64, W](s[offset + j])
                            * result_scalar
                        ) / s0,
                    )

                for left in range(n):
                    var entry = SIMD[DType.float64, W](0.0)
                    for j in range(size):
                        entry += (
                            SIMD[DType.float64, W](
                                g[(offset + j) * n + left]
                            ) * work.load[width=W](j * W)
                        )
                    matrix.store(
                        left * dim + right,
                        matrix.load[width=W](left * dim + right) + entry,
                    )
                right += W

            while right < n:
                var g0 = g[offset * n + right]
                var product_scalar = z[offset] * g0
                for j in range(1, size):
                    product_scalar += (
                        z[offset + j] * g[(offset + j) * n + right]
                    )
                    product[j] = (
                        z[offset + j] * g0
                        + z[offset] * g[(offset + j) * n + right]
                    )
                product[0] = product_scalar
                lorentz_solve(s + offset, product, work, size)
                for left in range(n):
                    var entry = 0.0
                    for j in range(size):
                        entry += g[(offset + j) * n + left] * work[j]
                    matrix[left * dim + right] += entry
                right += 1
        offset += size


def recover_dz_soc(
    g: Ptr,
    s: Ptr,
    z: Ptr,
    raw: Ptr,
    q: IPtr,
    work: Ptr,
    dx: Ptr,
    dz: Ptr,
    n: Int,
    m: Int,
    l: Int,
    nq: Int,
):
    for row in range(l):
        dz[row] = (
            raw[row] + z[row] * dot(g + row * n, dx, n)
        ) / s[row]
    var product = work + m
    var offset = l
    for cone in range(nq):
        var size = Int(q[cone])
        for j in range(size):
            work[offset + j] = dot(g + (offset + j) * n, dx, n)
        var product_scalar = z[offset] * work[offset]
        for j in range(1, size):
            product_scalar += z[offset + j] * work[offset + j]
            product[j] = (
                z[offset + j] * work[offset]
                + z[offset] * work[offset + j]
            )
        product[0] = product_scalar
        for j in range(size):
            product[j] += raw[offset + j]
        lorentz_solve(s + offset, product, dz + offset, size)
        offset += size


def gaussian_factor(matrix: Ptr, pivots: IPtr, dim: Int) -> Bool:
    for pivot in range(dim):
        var best_row = pivot
        var best = abs(matrix[pivot * dim + pivot])
        for row in range(pivot + 1, dim):
            var value = abs(matrix[row * dim + pivot])
            if value > best:
                best = value
                best_row = row
        if best < 1e-13:
            return False
        pivots[pivot] = Int64(best_row)
        if best_row != pivot:
            var pivot_row = matrix + pivot * dim
            var swap_row = matrix + best_row * dim
            var col = 0
            while col + W <= dim:
                var tmp = pivot_row.load[width=W](col)
                pivot_row.store(col, swap_row.load[width=W](col))
                swap_row.store(col, tmp)
                col += W
            while col < dim:
                var tmp = pivot_row[col]
                pivot_row[col] = swap_row[col]
                swap_row[col] = tmp
                col += 1

        var diagonal = matrix[pivot * dim + pivot]

        @parameter
        def eliminate(row: Int):
            var factor = matrix[row * dim + pivot] / diagonal
            matrix[row * dim + pivot] = factor
            if factor != 0.0:
                var factors = SIMD[DType.float64, W](factor)
                var col = pivot + 1
                while col + W <= dim:
                    matrix.store(
                        row * dim + col,
                        matrix.load[width=W](row * dim + col)
                        - factors
                        * matrix.load[width=W](pivot * dim + col),
                    )
                    col += W
                while col < dim:
                    matrix[row * dim + col] -= factor * matrix[pivot * dim + col]
                    col += 1

        var remaining = dim - pivot - 1
        if remaining * remaining >= PARALLEL_ELIMINATION_WORK:
            @parameter
            def eliminate_offset(index: Int):
                eliminate(pivot + 1 + index)

            parallelize[eliminate_offset](remaining, PARALLEL_WORKERS)
        else:
            for row in range(pivot + 1, dim):
                eliminate(row)
    return True


def gaussian_solve_factored(
    matrix: Ptr,
    pivots: IPtr,
    rhs: Ptr,
    solution: Ptr,
    dim: Int,
):
    for pivot in range(dim):
        var swap_row = Int(pivots[pivot])
        if swap_row != pivot:
            var tmp = rhs[pivot]
            rhs[pivot] = rhs[swap_row]
            rhs[swap_row] = tmp

    for row in range(dim):
        var values = SIMD[DType.float64, W](0.0)
        var col = 0
        while col + W <= row:
            values += (
                matrix.load[width=W](row * dim + col)
                * rhs.load[width=W](col)
            )
            col += W
        var value = rhs[row] - values.reduce_add()
        while col < row:
            value -= matrix[row * dim + col] * rhs[col]
            col += 1
        rhs[row] = value

    for rev in range(dim):
        var row = dim - 1 - rev
        var values = SIMD[DType.float64, W](0.0)
        var col = row + 1
        while col + W <= dim:
            values += (
                matrix.load[width=W](row * dim + col)
                * solution.load[width=W](col)
            )
            col += W
        var value = rhs[row] - values.reduce_add()
        while col < dim:
            value -= matrix[row * dim + col] * solution[col]
            col += 1
        solution[row] = value / matrix[row * dim + row]


def recover_ds(g: Ptr, rg: Ptr, dx: Ptr, ds: Ptr, n: Int, m: Int):
    @parameter
    def compute_row(row: Int):
        ds[row] = -rg[row] - dot(g + row * n, dx, n)

    if m * n >= PARALLEL_MATVEC_WORK:
        parallelize[compute_row](m, PARALLEL_WORKERS)
    else:
        for row in range(m):
            compute_row(row)


def solve_impl(
    c: Ptr,
    g: Ptr,
    h: Ptr,
    a: Ptr,
    b: Ptr,
    q: IPtr,
    pivots: IPtr,
    x: Ptr,
    y: Ptr,
    z: Ptr,
    s: Ptr,
    matrix: Ptr,
    rhs: Ptr,
    direction: Ptr,
    ds_aff: Ptr,
    dz_aff: Ptr,
    rd: Ptr,
    rp: Ptr,
    rg: Ptr,
    correction: Ptr,
    stats: Ptr,
    n: Int,
    m: Int,
    p: Int,
    l: Int,
    nq: Int,
    max_iters: Int,
    abstol: Float64,
    reltol: Float64,
    feastol: Float64,
) -> Int:
    var dim = n + p + m
    var linear_dim = n + p
    var degree = l + 2 * nq
    if degree <= 0:
        return -2

    var zeros = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        x.store(i, zeros)
        i += W
    while i < n:
        x[i] = 0.0
        i += 1
    i = 0
    while i + W <= p:
        y.store(i, zeros)
        i += W
    while i < p:
        y[i] = 0.0
        i += 1
    cone_identity(s, l, q, nq)
    cone_identity(z, l, q, nq)

    var cscale = 1.0 + norm_inf(c, n)
    var bscale = 1.0 + norm_inf(b, p)
    var hscale = 1.0 + norm_inf(h, m)
    var status = -7
    var completed = 0

    for iteration in range(max_iters + 1):
        residuals(c, g, h, a, b, x, y, z, s, rd, rp, rg, n, m, p)
        var primal_residual = max(norm_inf(rp, p) / bscale, norm_inf(rg, m) / hscale)
        var dual_residual = norm_inf(rd, n) / cscale
        var gap = cone_dot(s, z, m)
        var pcost = dot(c, x, n)
        var dcost = -dot(h, z, m) - dot(b, y, p)
        stats[1] = primal_residual
        stats[2] = dual_residual
        stats[3] = gap
        stats[4] = pcost
        stats[5] = dcost

        if (
            primal_residual <= feastol
            and dual_residual <= feastol
            and gap <= abstol + reltol * max(abs(pcost), abs(dcost))
        ):
            status = 0
            completed = iteration
            break
        if iteration == max_iters:
            completed = iteration
            break

        var mu = gap / Float64(degree)
        if nq == 0:
            fill_newton_linear(
                g, a, s, z, rd, rp, rg, correction, matrix, rhs,
                n, m, p, 0.0, False, True,
            )
            if not gaussian_factor(matrix, pivots, linear_dim):
                status = -2
                completed = iteration
                break
            gaussian_solve_factored(
                matrix, pivots, rhs, direction, linear_dim
            )
            recover_dz_linear(
                g, s, z, rhs + linear_dim, direction,
                direction + n + p, n, m,
            )
        else:
            fill_newton_reduced_soc(
                g, a, s, z, rd, rp, rg, correction, matrix, rhs, q,
                n, m, p, l, nq, 0.0, False, True,
            )
            if not gaussian_factor(matrix, pivots, linear_dim):
                status = -2
                completed = iteration
                break
            gaussian_solve_factored(
                matrix, pivots, rhs, direction, linear_dim
            )
            recover_dz_soc(
                g, s, z, rhs + linear_dim, q,
                matrix + linear_dim * linear_dim,
                direction, direction + n + p, n, m, l, nq,
            )
        recover_ds(g, rg, direction, ds_aff, n, m)
        i = 0
        while i + W <= m:
            dz_aff.store(i, direction.load[width=W](n + p + i))
            i += W
        while i < m:
            dz_aff[i] = direction[n + p + i]
            i += 1

        var alpha_p_aff = cone_step(s, ds_aff, l, q, nq)
        var alpha_d_aff = cone_step(z, dz_aff, l, q, nq)
        var gap_values = SIMD[DType.float64, W](0.0)
        i = 0
        while i + W <= m:
            gap_values += (
                s.load[width=W](i)
                + SIMD[DType.float64, W](alpha_p_aff)
                * ds_aff.load[width=W](i)
            ) * (
                z.load[width=W](i)
                + SIMD[DType.float64, W](alpha_d_aff)
                * dz_aff.load[width=W](i)
            )
            i += W
        var gap_aff = gap_values.reduce_add()
        while i < m:
            gap_aff += (
                (s[i] + alpha_p_aff * ds_aff[i])
                * (z[i] + alpha_d_aff * dz_aff[i])
            )
            i += 1
        var ratio = max(0.0, gap_aff / Float64(degree)) / mu
        var sigma = min(1.0, ratio * ratio * ratio)
        cone_product(ds_aff, dz_aff, correction, l, q, nq)

        if nq == 0:
            fill_newton_linear(
                g, a, s, z, rd, rp, rg, correction, matrix, rhs,
                n, m, p, sigma * mu, True, False,
            )
            gaussian_solve_factored(
                matrix, pivots, rhs, direction, linear_dim
            )
            recover_dz_linear(
                g, s, z, rhs + linear_dim, direction,
                direction + n + p, n, m,
            )
        else:
            fill_newton_reduced_soc(
                g, a, s, z, rd, rp, rg, correction, matrix, rhs, q,
                n, m, p, l, nq, sigma * mu, True, False,
            )
            gaussian_solve_factored(
                matrix, pivots, rhs, direction, linear_dim
            )
            recover_dz_soc(
                g, s, z, rhs + linear_dim, q,
                matrix + linear_dim * linear_dim,
                direction, direction + n + p, n, m, l, nq,
            )
        recover_ds(g, rg, direction, rhs, n, m)

        var alpha_p = min(1.0, 0.99 * cone_step(s, rhs, l, q, nq))
        var alpha_d = min(
            1.0, 0.99 * cone_step(z, direction + n + p, l, q, nq)
        )
        var alpha_p_values = SIMD[DType.float64, W](alpha_p)
        var alpha_d_values = SIMD[DType.float64, W](alpha_d)
        i = 0
        while i + W <= n:
            x.store(
                i,
                x.load[width=W](i)
                + alpha_p_values * direction.load[width=W](i),
            )
            i += W
        while i < n:
            x[i] += alpha_p * direction[i]
            i += 1
        i = 0
        while i + W <= m:
            s.store(
                i,
                s.load[width=W](i)
                + alpha_p_values * rhs.load[width=W](i),
            )
            z.store(
                i,
                z.load[width=W](i)
                + alpha_d_values
                * direction.load[width=W](n + p + i),
            )
            i += W
        while i < m:
            s[i] += alpha_p * rhs[i]
            z[i] += alpha_d * direction[n + p + i]
            i += 1
        i = 0
        while i + W <= p:
            y.store(
                i,
                y.load[width=W](i)
                + alpha_d_values * direction.load[width=W](n + i),
            )
            i += W
        while i < p:
            y[i] += alpha_d * direction[n + i]
            i += 1
        completed = iteration + 1

    stats[0] = Float64(completed)
    stats[6] = Float64(status)
    return status


@export("mecos_solve")
def mecos_solve(
    c_addr: Int,
    g_addr: Int,
    h_addr: Int,
    a_addr: Int,
    b_addr: Int,
    q_addr: Int,
    pivots_addr: Int,
    x_addr: Int,
    y_addr: Int,
    z_addr: Int,
    s_addr: Int,
    matrix_addr: Int,
    rhs_addr: Int,
    direction_addr: Int,
    ds_aff_addr: Int,
    dz_aff_addr: Int,
    rd_addr: Int,
    rp_addr: Int,
    rg_addr: Int,
    correction_addr: Int,
    stats_addr: Int,
    n: Int,
    m: Int,
    p: Int,
    l: Int,
    nq: Int,
    max_iters: Int,
    abstol: Float64,
    reltol: Float64,
    feastol: Float64,
) abi("C") -> Int:
    if (
        c_addr == 0 or g_addr == 0 or h_addr == 0 or a_addr == 0
        or b_addr == 0 or q_addr == 0 or pivots_addr == 0
        or x_addr == 0 or y_addr == 0 or z_addr == 0 or s_addr == 0
        or matrix_addr == 0 or rhs_addr == 0 or direction_addr == 0
        or ds_aff_addr == 0 or dz_aff_addr == 0 or rd_addr == 0
        or rp_addr == 0 or rg_addr == 0 or correction_addr == 0
        or stats_addr == 0
        or n <= 0 or m <= 0 or p < 0 or l < 0 or nq < 0
        or max_iters < 0 or l > m
    ):
        return -3
    var q_pointer = IPtr(unsafe_from_address=q_addr)
    var described_rows = l
    for cone in range(nq):
        var size = Int(q_pointer[cone])
        if size < 2 or size > m - described_rows:
            return -3
        described_rows += size
    if described_rows != m:
        return -3
    return solve_impl(
        Ptr(unsafe_from_address=c_addr),
        Ptr(unsafe_from_address=g_addr),
        Ptr(unsafe_from_address=h_addr),
        Ptr(unsafe_from_address=a_addr),
        Ptr(unsafe_from_address=b_addr),
        q_pointer,
        IPtr(unsafe_from_address=pivots_addr),
        Ptr(unsafe_from_address=x_addr),
        Ptr(unsafe_from_address=y_addr),
        Ptr(unsafe_from_address=z_addr),
        Ptr(unsafe_from_address=s_addr),
        Ptr(unsafe_from_address=matrix_addr),
        Ptr(unsafe_from_address=rhs_addr),
        Ptr(unsafe_from_address=direction_addr),
        Ptr(unsafe_from_address=ds_aff_addr),
        Ptr(unsafe_from_address=dz_aff_addr),
        Ptr(unsafe_from_address=rd_addr),
        Ptr(unsafe_from_address=rp_addr),
        Ptr(unsafe_from_address=rg_addr),
        Ptr(unsafe_from_address=correction_addr),
        Ptr(unsafe_from_address=stats_addr),
        n, m, p, l, nq, max_iters, abstol, reltol, feastol,
    )
