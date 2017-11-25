<?php
$myfile = fopen("resources/test.csv", "w");

$list = array(
	array($_POST["movie_id_1"], $_POST["ratings_1"]),
	array($_POST["movie_id_2"], $_POST["ratings_2"])
);
$txt = " ".$_POST["name"]."\t".$_POST["ratings"]."\n";
echo var_dump($txt);
foreach ($list as $fields) {
	fputcsv($myfile, $fields);
}


//fwrite($myfile, $txt);

//fwrite($myfile, "\t".$_POST["ratings"]);
fclose($myfile);
?>