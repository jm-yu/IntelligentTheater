<?php
$servername = "localhost";
$username = "root";
$password = "root";
try{
	$conn = new PDO("mysql:host=$servername;dbname=movie", $username, $password);
	$conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
}
catch(PDOException $e){
	echo $sql."<br>".$e->getMessage();
}
function getID($movie_name, $conn){
	try{
		$sql = "SELECT movie_id FROM movie_info_test WHERE movie_name = '".$movie_name."'";
		//echo $sql."<br>";
		$query = $conn->prepare($sql);
		$query->execute();
		$list = $query->fetchAll();
		//echo $list[0][0]."<br>";
		return $list[0][0];
	}
	catch(PDOException $e){
		echo $sql."<br>".$e->getMessage();
	}
}
$myfile = fopen("resources/test.csv", "w");

$list = array(
	array(getID(addslashes($_POST["movie_id_1"]),$conn), $_POST["ratings_1"]),
	array(getID(addslashes($_POST["movie_id_2"]),$conn), $_POST["ratings_2"]),
	array(getID(addslashes($_POST["movie_id_3"]),$conn), $_POST["ratings_3"]),
	array(getID(addslashes($_POST["movie_id_4"]),$conn), $_POST["ratings_4"]),
	array(getID(addslashes($_POST["movie_id_5"]),$conn), $_POST["ratings_5"])
);
//echo getID(123, $conn);
$txt = " ".$_POST["name"]."\t".$_POST["ratings"]."\n";

foreach ($list as $fields) {
	fputcsv($myfile, $fields);
}
fclose($myfile);
sleep(3);
$output = fopen("resources/rank.txt", "r");
$ranks = fgets($output);
$ranks = str_replace("[", "", $ranks);
$ranks = str_replace("]", "", $ranks);
$ranks = explode(",", $ranks);
foreach ($ranks as $rank){
	try{
		$sql = "SELECT movie_name FROM movie_info_test WHERE movie_id = '".trim($rank)."'";
		//echo $sql."<br>";
		$query = $conn->prepare($sql);
		$query->execute();
		$list = $query->fetchAll();
		echo $list[0][0]."<br>";
	}
	catch(PDOException $e){
		echo $sql."<br>".$e->getMessage();
	}
}

fclose($output);


//fwrite($myfile, $txt);

//fwrite($myfile, "\t".$_POST["ratings"]);
?>