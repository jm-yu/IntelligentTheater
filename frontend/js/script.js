// autocomplet : this function will be executed every time we change the text

$(document).ready(function(){
    
	$('#movie_id_1, #movie_id_2').keyup(function(){
		var min_length = 0; // min caracters to display the autocomplete
		var keyword = $(this).val();
		var movie_id = jQuery(this).attr("id");
		var list_id = "#list_" + movie_id;
	//alert(list_id);
		if (keyword.length >= min_length) {
			$.ajax({
				url: 'ajax_refresh.php',
				type: 'POST',
				data: {keyword:keyword},
				success:function(data){
					$(list_id).show();
					$(list_id).html(data);
				}
			});
		} else {
			$(list_id).hide();
		}
	});
	$('#lim').onclick(function(){
		alert("1");
		$('#movie_id_1').val(item);
			// hide proposition list
		$('#list_move_id_1').hide();
	});
});
/*function autocomplet() {
	var min_length = 0; // min caracters to display the autocomplete
	var keyword = $('#movie_id_2').val();
	if (keyword.length >= min_length) {
		$.ajax({
			url: 'ajax_refresh.php',
			type: 'POST',
			data: {keyword:keyword},
			success:function(data){
				$('#list_move_id_2').show();
				$('#list_move_id_2').html(data);
			}
		});
	} else {
		$('#list_move_id_2').hide();
	}
}*/

// set_item : this function will be executed when we select an item
function set_item(item, id1) {
	// change input value
	$('#movie_id_1').val(item);
	alert(id1);
	// hide proposition list
	$('#list_move_id_1').hide();

}
