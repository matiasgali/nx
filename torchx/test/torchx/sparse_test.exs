defmodule Torchx.SparseTest do
  use Torchx.Case

  setup do
    indices = Nx.tensor([[0], [3]])

    t1_dense =
      Nx.tensor([1, 0, 0, 2])

    t1 =
      t1_dense
      |> Nx.gather(indices)
      |> Torchx.from_nx()
      |> Torchx.to_sparse(Torchx.from_nx(Nx.transpose(indices)), {4})
      |> Torchx.to_nx()

    t2_dense =
      Nx.tensor([3, 0, 0, 4])

    t2 =
      t2_dense
      |> Nx.gather(indices)
      |> Torchx.from_nx()
      |> Torchx.to_sparse(Torchx.from_nx(Nx.transpose(indices)), {4})
      |> Torchx.to_nx()

    %{t1: t1, t1_dense: t1_dense, t2: t2, t2_dense: t2_dense}
  end

  for bin_op <- [:add, :subtract, :multiply, :divide] do
    test "#{bin_op}", %{t1: t1, t1_dense: t1_dense, t2: t2, t2_dense: t2_dense} do
      sparse_add = Nx.unquote(bin_op)(t1, t2)

      sparse_add
      |> Torchx.from_nx()
      |> Torchx.to_dense()
      |> Torchx.to_nx()
      |> assert_equal(Nx.unquote(bin_op)(t1_dense, t2_dense))
    end
  end

  for unary_op <- [:any, :asin, :is_nan, :negate, :sqrt, :transpose] do
    test "#{unary_op}", %{t1: t1, t1_dense: t1_dense} do
      sparse_add = Nx.unquote(unary_op)(t1)

      sparse_add
      |> Torchx.from_nx()
      |> Torchx.to_dense()
      |> Torchx.to_nx()
      |> assert_equal(Nx.unquote(unary_op)(t1_dense))
    end
  end

  test "shape", %{t1: t1, t1_dense: t1_dense} do
    assert Nx.shape(t1) == Nx.shape(t1_dense)
  end


  # test "gather", %{t1: t1, t1_dense: t1_dense} do
  #   # fails due to reshape being unavailable
  #   sparse = Nx.gather(t1, Nx.tensor([[0], [0], [1], [1], [2], [2], [3], [3]]))
  #   dense = Nx.gather(t1_dense, Nx.tensor([[0], [0], [1], [1], [2], [2], [3], [3]]))

  #   sparse
  #   |> Torchx.from_nx()
  #   |> Torchx.to_dense()
  #   |> Torchx.to_nx()
  #   |> assert_equal(dense)
  # end


  test "log"
  test "log1p"
  test "square"
  test "pow"
end
