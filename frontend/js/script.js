// autocomplet : this function will be executed every time we change the text
function autocomplet() {
	var min_length = 0; // min caracters to display the autocomplete
	var keyword = $.trim($('#movie_id').val());
	if (keyword.length >= min_length) {
		$.ajax({
			url: 'ajax_refresh.php',
			type: 'POST',
			data: {keyword:keyword},
			success:function(data){
				$('#movie_list_id').show();
				$('#movie_list_id').html(data);
			}
		});
	} else {
		$('#movie_list_id').hide();
	}
}

// set_item : this function will be executed when we select an item
function set_item(item) {
	// change input value
	$('#movie_id').val(item);
	// hide proposition list
	$('#movie_list_id').hide();
}
//var table = document.getElementById('movie_list_id'),
//    selected = table.getElementsByClassName('selected');
//table.onclick = highlight;