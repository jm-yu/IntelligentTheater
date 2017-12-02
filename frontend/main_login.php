<?php
    session_start();
    if (isset($_SESSION['username'])) {
        header("./index.html");
   }
?>
<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/html">
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="css/login.css" />
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
    <script src="js/login_script.js"></script>

</head>
<body>
<!-- Button to open the modal login form -->
<h2>Movie recommendation</h2>
<span align="left">
<button onclick="document.getElementById('id01').style.display='block'">Login</button>
</span>
<span align="right">
<button onclick="document.getElementById('id01').style.display='block'">new user</button>
</span>
<!-- The Modal -->
<div id="id01" class="modal">
  <span onclick="document.getElementById('id01').style.display='none'" 
class="close" title="Close Modal">&times;</span>

  <!-- Modal Content -->
  <form class="modal-content animate"">
    <div class="imgcontainer">
      <img src="img_avatar2.png" alt="Avatar" class="avatar">
    </div>

    <div class="container">
      <label><b>Username</b></label>
      <input type="text" id="myusername" placeholder="Enter Username" name="uname" required>

      <label><b>Password</b></label>
      <input id="mypassword" type="password" placeholder="Enter Password" name="psw" required>

      <button id = "submit" type="submit">Login</button>
    </div>

    <div class="container" style="background-color:#f1f1f1">
      <button type="button" onclick="document.getElementById('id01').style.display='none'" class="cancelbtn">Cancel</button>
    </div>
  </form>
</div>
<div>
<a href="signup.php" name="Sign Up" id="signup" class="btn btn-lg btn-primary btn-block" type="submit">Create new account</a>
<div>
<div id="message"></div>
</body>
</html>