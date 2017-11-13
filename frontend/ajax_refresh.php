<?php
// PDO connect *********
define ('DBUSER', 'root');
define ('DBPASS', 'root');
define ('DBNAME', 'movie');

function connect() {
    return new PDO('mysql:host=localhost;dbname=movie;port=8889', 'root', 'root', array(PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION, PDO::MYSQL_ATTR_INIT_COMMAND => "SET NAMES utf8"));
}

$pdo = connect();
$keyword = $_POST['keyword'];
$sql = "SELECT * FROM outputtmp WHERE starting_phrase like (:keyword) ORDER BY count DESC LIMIT 0, 10";
$query = $pdo->prepare($sql);
$query->bindParam(':keyword', $keyword, PDO::PARAM_STR);
$query->execute();
$list = $query->fetchAll();
foreach ($list as $rs) {
	// put in bold the written text
	//$movie_name = str_replace($_POST['keyword'], '<b>'.$_POST['keyword'].'</b>', $rs['following_word']);
	$predictor = $rs['following_word'];
	// add new option
    echo '<li onclick="set_item(\''.str_replace("'", "\'", $rs['following_word']).'\')">'.$predictor.'</li>';
    //echo '<li onclick="set_item(\''.str_replace("'", "\'", $rs['following_word']).'\')">'.$keyword.'</li>';
}
?>
