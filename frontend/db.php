<?php
$servername = "localhost";
$username = "root";
$password = "root";


try{
	$conn = new PDO("mysql:host=$servername;dbname=movie", $username, $password);
	$conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
	//$sql = "CREATE DATABASE myDB";
	//$conn->exec($sql);
	//$conn->exec("select * from outputtmp where starting_phrase like 'a' ");
	
	$myfile = fopen("resources/m.csv", "r") or die("unable to open file");
	$data = fgetcsv($myfile, 1000, ",");
	$data = fgetcsv($myfile, 1000, ",");
	while (($data = fgetcsv($myfile, 1000, ",")) != FALSE){
		$id = $data[0];
		$name = $data[1];
		echo $id."\t".$name."\n\n";
		//echo var_dump($id);
	}
	fclose($myfile);
	/*$sql = "SELECT * FROM movie_info";
	$query = $conn->prepare($sql);
	//$query->bindParam(':keyword', $keyword, PDO::PARAM_STR);
	$query->execute();
	$list = $query->fetchAll();
	echo var_dump($list);*/
	echo "success1";
}
catch(PDOException $e){
	echo $sql."<br>".$e->getMessage();
}
?>
