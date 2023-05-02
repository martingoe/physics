use nalgebra::{Dyn, OMatrix, OVector, U1};

pub struct SparseMatrix {
    blocks: Vec<SparseMatrixBlock>,
    nrows: usize,
    ncols: usize,
}
impl SparseMatrix{
    pub fn new(nrows: usize, ncols: usize) -> Self{
        Self{
            blocks: vec![],
            nrows,
            ncols,
        }
    }
    pub fn add_block(&mut self, row: usize, column: usize, nrows: usize, ncols: usize, data: OMatrix<f32, Dyn, Dyn>){
        self.blocks.push(SparseMatrixBlock{
            i: row,
            j: column,
            i_length: nrows,
            j_length: ncols,
            data,
        })
    }
    pub fn multiply_vector(&self, vector: &OMatrix<f32, Dyn, U1>) -> OVector<f32, Dyn>{
        assert_eq!(self.ncols, vector.nrows());
        let mut res = OVector::<f32, Dyn>::repeat(self.nrows, 0.0);
        for block in &self.blocks{
            for row in 0..block.i_length {
                let mut result = OVector::<f32, Dyn>::zeros(1);
                block.data.row(row).mul_to(&vector.rows(block.j, block.j_length), &mut result);
                res[(block.i + row, 0)] += result[(0, 0)];

            }
        }
        res
    }

    pub fn tr_multiply_vector(&self, vector: &OMatrix<f32, Dyn, U1>) -> OVector<f32, Dyn>{
        assert_eq!(self.nrows, vector.nrows());
        let mut res = OVector::<f32, Dyn>::repeat(self.ncols, 0.0);
        for block in &self.blocks{
            for col in 0..block.j_length {
                let mut result = OVector::<f32, Dyn>::zeros(1);
                block.data.column(col).transpose().mul_to(&vector.rows(block.i, block.i_length), &mut result);
                res[(block.j + col, 0)] += result[(0, 0)];
            }
        }
        res
    }
}
struct SparseMatrixBlock {
    i: usize,
    j: usize,
    i_length: usize,
    j_length: usize,
    data: OMatrix<f32, Dyn, Dyn>
}

#[cfg(test)]
mod test{
    use nalgebra::{Dyn, OMatrix, OVector, RowVector};
    use crate::physics::sparse_matrix::SparseMatrix;

    #[test]
    fn multiply_vector_single_block_test(){
        let mut sparse_matrix = SparseMatrix::new(5, 5);
        sparse_matrix.add_block(
            0,
            0,
            2,
            2,
            OMatrix::<f32, Dyn, Dyn>::from_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]));
        let vector = OVector::from(vec![2.0, 6.0, 1.0, 1.0, 2.0]);

        let expected = OVector::from(vec![14.0, 30.0, 0.0, 0.0, 0.0]);
        assert_eq!(sparse_matrix.multiply_vector(&vector), expected);

    }
    #[test]
    fn multiply_vector_multiple_block_test(){
        let mut sparse_matrix = SparseMatrix::new(5, 5);
        sparse_matrix.add_block(
            0,
            0,
            2,
            2,
            OMatrix::<f32, Dyn, Dyn>::from_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]));
        sparse_matrix.add_block(
            1,
            2,
            2,
            1,
            OMatrix::<f32, Dyn, Dyn>::from_vec(2, 1, vec![1.0, 3.0]));
        let vector = OVector::from(vec![2.0, 6.0, 1.0, 1.0, 2.0]);

        let expected = OVector::from(vec![14.0, 31.0, 3.0, 0.0, 0.0]);
        assert_eq!(sparse_matrix.multiply_vector(&vector), expected);
    }
    #[test]
    fn tr_multiply_vector_multiple_block_test(){
        let mut sparse_matrix = SparseMatrix::new(5, 5);
        sparse_matrix.add_block(
            0,
            0,
            2,
            2,
            OMatrix::<f32, Dyn, Dyn>::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]));
        sparse_matrix.add_block(
            2,
            1,
            1,
            2,
            OMatrix::<f32, Dyn, Dyn>::from_vec(1, 2, vec![1.0, 3.0]));
        let vector = OVector::from(vec![2.0, 6.0, 1.0, 1.0, 2.0]);

        let expected = OVector::from(vec![14.0, 31.0, 3.0, 0.0, 0.0]);
        assert_eq!(sparse_matrix.tr_multiply_vector(&vector), expected);
    }
}