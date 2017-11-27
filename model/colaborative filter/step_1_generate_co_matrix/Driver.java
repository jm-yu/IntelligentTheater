
public class Driver {
	public static void main(String[] args) throws Exception {

		DataDividerByUser dataDividerByUser = new DataDividerByUser();
		CoOccurrenceMatrixGenerator coOccurrenceMatrixGenerator = new CoOccurrenceMatrixGenerator();
		Normalize normalize = new Normalize();
		Multiplication multiplication = new Multiplication();
		Sum sum = new Sum();

		String rawInput = args[0];
		String userMovieListOutputDir = "tmp1/";
		String coOccurrenceMatrixDir = "temp/";
		String normalizeDir = args[1];

		String[] path1 = {rawInput, userMovieListOutputDir};
		String[] path2 = {userMovieListOutputDir, coOccurrenceMatrixDir};
		String[] path3 = {coOccurrenceMatrixDir, normalizeDir};


		dataDividerByUser.main(path1);
		coOccurrenceMatrixGenerator.main(path2);
		normalize.main(path3);

	}

}
