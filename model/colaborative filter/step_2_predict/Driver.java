
public class Driver {
    public static void main(String[] args) throws Exception {
        
        DataDividerByUser dataDividerByUser = new DataDividerByUser();
        CoOccurrenceMatrixGenerator coOccurrenceMatrixGenerator = new CoOccurrenceMatrixGenerator();
        Normalize normalize = new Normalize();
        Multiplication multiplication = new Multiplication();
        Sum sum = new Sum();
        
        String one_user_rating = args[0];
        String normalizeDir = args[1];
        String multiplicationDir = "tmp3/";
        String sumDir = args[2];
        String[] path4 = {normalizeDir, one_user_rating, multiplicationDir};
        String[] path5 = {multiplicationDir, sumDir};
        
        multiplication.main(path4);
        sum.main(path5);
        
    }
    
}

