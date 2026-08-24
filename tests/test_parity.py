import inspect

import ecos
import numpy as np
import pytest
from scipy import sparse

import mojo_ecos
from mojo_ecos._lib import lib


def run_both(c, G, h, dims, A=None, b=None, **kwargs):
    Gs = sparse.csc_matrix(np.asarray(G, dtype=float))
    As = None if A is None else sparse.csc_matrix(np.asarray(A, dtype=float))
    options = {"verbose": False, **kwargs}
    reference = ecos.solve(c, Gs, h, dims, As, b, **options)
    result = mojo_ecos.solve(c, Gs, h, dims, As, b, **options)
    return reference, result


def assert_optimal_parity(reference, result, *, x_atol=2e-5, obj_atol=2e-7):
    assert reference["info"]["exitFlag"] == 0
    assert result["info"]["exitFlag"] == 0
    assert result["info"]["pcost"] == pytest.approx(
        reference["info"]["pcost"], abs=obj_atol, rel=2e-7
    )
    assert result["x"] == pytest.approx(reference["x"], abs=x_atol, rel=2e-5)
    assert result["info"]["pres"] <= 2e-8
    assert result["info"]["dres"] <= 2e-8


def box_lp(n, seed):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    G = np.vstack((np.eye(n), -np.eye(n)))
    h = np.full(2 * n, 2.0)
    return c, G, h, {"l": 2 * n, "q": []}


@pytest.mark.parametrize("n,seed", [(1, 1), (3, 2), (8, 3), (15, 4)])
def test_box_lp_matches_upstream(n, seed):
    reference, result = run_both(*box_lp(n, seed))
    assert_optimal_parity(reference, result)


def test_simd_tail_dimensions_match_upstream():
    reference, result = run_both(*box_lp(7, 21))
    assert_optimal_parity(reference, result)


def test_zero_copy_single_entry_rows_match_upstream():
    reference, result = run_both(*box_lp(80, 23))
    assert_optimal_parity(reference, result)


def test_single_entry_rows_with_one_equality_match_upstream():
    c, G, h, dims = box_lp(9, 24)
    A = np.ones((1, 9))
    b = np.array([0.5])
    reference, result = run_both(c, G, h, dims, A, b)
    assert_optimal_parity(reference, result)


def test_multi_entry_row_uses_dense_fallback():
    c = np.array([-1.0, -0.5])
    G = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([1.0, 1.0, 1.0])
    reference, result = run_both(c, G, h, {"l": 3, "q": []})
    assert_optimal_parity(reference, result)


def test_lp_with_equalities_matches_upstream():
    c = np.array([-3.0, 1.0, 2.0, -1.0])
    G = np.vstack((np.eye(4), -np.eye(4)))
    h = np.ones(8)
    A = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    b = np.array([0.25, -0.4])
    reference, result = run_both(c, G, h, {"l": 8, "q": []}, A, b)
    assert_optimal_parity(reference, result)
    assert result["y"] == pytest.approx(reference["y"], abs=2e-6)


@pytest.mark.parametrize("point", [(3.0, 4.0), (-2.0, 5.0), (0.25, -0.75)])
def test_second_order_cone_norm_matches_upstream(point):
    c = np.array([1.0, 0.0, 0.0])
    G = -np.eye(3)
    h = np.zeros(3)
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    b = np.asarray(point)
    reference, result = run_both(c, G, h, {"l": 0, "q": [3]}, A, b)
    assert_optimal_parity(reference, result, x_atol=1e-6)
    assert result["x"][0] == pytest.approx(np.linalg.norm(point), abs=2e-7)
    assert result["z"] == pytest.approx(reference["z"], abs=2e-6)


def portfolio_problem(n=8):
    weights = np.linspace(0.5, 2.0, n)
    c = np.r_[1.0, np.zeros(n)]
    G_linear = np.c_[np.zeros((n, 1)), -np.eye(n)]
    G_soc = np.zeros((n + 1, n + 1))
    G_soc[0, 0] = -1.0
    G_soc[1:, 1:] = -np.diag(weights)
    G = np.vstack((G_linear, G_soc))
    h = np.zeros(2 * n + 1)
    A = np.c_[np.zeros((1, 1)), np.ones((1, n))]
    b = np.array([1.0])
    return c, G, h, {"l": n, "q": [n + 1]}, A, b


@pytest.mark.parametrize("n", [3, 8, 15])
def test_mixed_linear_and_soc_portfolio_matches_upstream(n):
    reference, result = run_both(*portfolio_problem(n))
    assert_optimal_parity(reference, result, x_atol=5e-5)
    assert np.min(result["x"][1:]) >= -2e-8
    assert result["x"][1:].sum() == pytest.approx(1.0, abs=2e-8)


def test_multiple_soc_blocks_match_upstream():
    c = np.array([1.0, 1.0, 0.0, 0.0])
    G = np.zeros((6, 4))
    G[0, 0], G[1, 2], G[2, 3] = -1.0, -1.0, -1.0
    G[3, 1], G[4, 2], G[5, 3] = -1.0, -2.0, -1.0
    A = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    b = np.array([3.0, 4.0])
    reference, result = run_both(
        c, G, np.zeros(6), {"l": 0, "q": [3, 3]}, A, b
    )
    assert_optimal_parity(reference, result, x_atol=2e-6)
    assert result["x"][:2] == pytest.approx([5.0, np.sqrt(52.0)], abs=2e-7)


def test_return_schema_matches_upstream():
    reference, result = run_both(*box_lp(3, 11))
    assert set(result) == set(reference) == {"x", "y", "z", "s", "info"}
    assert set(result["info"]) == set(reference["info"])
    assert set(result["info"]["timing"]) == {"runtime", "tsetup", "tsolve"}
    assert result["x"].dtype == result["z"].dtype == result["s"].dtype == np.float64


def test_kkt_residuals_and_cone_membership():
    c, G, h, dims, A, b = portfolio_problem(10)
    result = mojo_ecos.solve(
        c, sparse.csc_matrix(G), h, dims, sparse.csc_matrix(A), b, verbose=False
    )
    x, y, z, s = (result[name] for name in ("x", "y", "z", "s"))
    assert np.linalg.norm(A @ x - b, np.inf) <= 2e-8
    assert np.linalg.norm(G @ x + s - h, np.inf) <= 2e-8
    assert np.linalg.norm(c + A.T @ y + G.T @ z, np.inf) <= 2e-8
    assert np.min(s[:10]) > 0.0 and np.min(z[:10]) > 0.0
    assert s[10] > np.linalg.norm(s[11:])
    assert z[10] > np.linalg.norm(z[11:])
    assert s @ z <= 1e-8


def test_csr_inputs_warn_and_are_converted():
    c, G, h, dims = box_lp(4, 12)
    with pytest.warns(UserWarning, match="Converting G"):
        result = mojo_ecos.solve(c, sparse.csr_matrix(G), h, dims, verbose=False)
    assert result["info"]["exitFlag"] == 0


def test_integer_inputs_are_coerced_to_float64():
    c = np.array([1])
    G = sparse.csc_matrix(np.array([[-1]]))
    result = mojo_ecos.solve(c, G, np.array([-1]), {"l": 1}, verbose=False)
    assert result["x"].dtype == np.float64
    assert result["x"][0] == pytest.approx(1.0, abs=2e-8)


def test_max_iterations_matches_ecos_exit_convention():
    c, G, h, dims = box_lp(5, 13)
    result = mojo_ecos.solve(
        c, sparse.csc_matrix(G), h, dims, verbose=False, max_iters=0
    )
    assert result["info"]["exitFlag"] == -7
    assert result["info"]["iter"] == 0
    assert "Maximum" in result["info"]["infostring"]


def test_verbose_and_tolerance_options_are_applied(capsys):
    c, G, h, dims = box_lp(2, 17)
    result = mojo_ecos.solve(
        c,
        sparse.csc_matrix(G),
        h,
        dims,
        verbose=True,
        abstol=1e-7,
        reltol=1e-7,
        feastol=1e-7,
    )
    assert result["info"]["exitFlag"] == 0
    assert "mojo-ecos: Optimal solution found" in capsys.readouterr().out


def test_large_single_entry_setup_with_zero_iterations():
    n = 257
    c, G, h, dims = box_lp(n, 22)
    result = mojo_ecos.solve(
        c,
        sparse.csc_matrix(G),
        h,
        dims,
        verbose=False,
        max_iters=0,
    )
    assert result["info"]["exitFlag"] == -7
    assert result["info"]["iter"] == 0
    assert np.isfinite(result["info"]["pres"])
    assert np.isfinite(result["info"]["dres"])


def test_signature_matches_upstream():
    assert str(inspect.signature(mojo_ecos.solve)) == str(inspect.signature(ecos.solve))


@pytest.mark.parametrize(
    "dims,error",
    [
        ({"l": 1, "e": 1}, NotImplementedError),
        ({"l": 0, "q": [1]}, ValueError),
        ({"l": 2}, ValueError),
    ],
)
def test_invalid_or_unsupported_cones_raise(dims, error):
    with pytest.raises(error):
        mojo_ecos.solve(
            np.array([1.0]),
            sparse.csc_matrix([[-1.0]]),
            np.array([-1.0]),
            dims,
            verbose=False,
        )


def test_upstream_sparse_matrix_requirement_is_preserved():
    with pytest.raises(TypeError, match="sparse matrix"):
        mojo_ecos.solve(
            np.array([1.0]), np.array([[-1.0]]), np.array([-1.0]), {"l": 1}
        )


def test_unknown_keyword_is_rejected():
    c, G, h, dims = box_lp(2, 7)
    with pytest.raises(TypeError, match="invalid keyword"):
        mojo_ecos.solve(
            c, sparse.csc_matrix(G), h, dims, verbose=False, made_up=True
        )


@pytest.mark.parametrize(
    "replacement,error",
    [
        ({"c": np.array([1.0 + 0.0j])}, TypeError),
        ({"c": np.array([1.0], dtype=np.longdouble)}, TypeError),
        ({"c": np.array([2**53 + 1], dtype=np.int64)}, ValueError),
        ({"dims": {"l": 2.0}}, TypeError),
        ({"max_iters": 1.5}, TypeError),
        ({"max_iters": 2**63}, OverflowError),
        ({"abstol": np.nan}, ValueError),
        ({"reltol": np.inf}, ValueError),
    ],
)
def test_lossy_or_invalid_numeric_inputs_are_rejected(replacement, error):
    c, G, h, dims = box_lp(1, 9)
    arguments = {
        "c": c,
        "G": sparse.csc_matrix(G),
        "h": h,
        "dims": dims,
        "verbose": False,
    }
    arguments.update(replacement)
    with pytest.raises(error):
        mojo_ecos.solve(**arguments)


def test_empty_objective_is_rejected_before_ffi():
    with pytest.raises(ValueError, match="must not be empty"):
        mojo_ecos.solve(
            np.empty(0),
            sparse.csc_matrix((1, 0)),
            np.ones(1),
            {"l": 1},
            verbose=False,
        )


def test_strided_inputs_are_copied_safely():
    base = np.arange(12.0)
    c = base[1:7:2]
    G = sparse.csc_matrix(-np.eye(3))
    h = base[2:8:2]
    result = mojo_ecos.solve(c, G, h, {"l": 3}, verbose=False)
    assert result["info"]["exitFlag"] == 0
    assert result["x"] == pytest.approx(-h, abs=2e-7)


def test_c_abi_rejects_null_addresses_without_dereferencing():
    assert lib().mecos_solve(*([0] * 31), 1.0, 1.0, 1.0) == -3
